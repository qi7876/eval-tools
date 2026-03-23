"""Command-line entrypoint."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import resolve_adapter
from .config import (
    JudgeConfig,
    load_build_chain_manifest_config,
    load_prepare_data_config,
    load_run_eval_config,
    load_validate_data_config,
)
from .constants import TASK_SPATIAL_IMAGINATION
from .dataset import scan_dataset, summarize_records
from .experiments import (
    build_chain_manifest,
    evaluate_oracle_chain,
    load_chain_pairs,
    load_prepared_for_protocol,
    summarize_evaluation_records,
    summarize_experiment_b,
)
from .judge import JudgeResponseFormatExhaustedError, OpenAIJudgeClient, StaticJudgeClient
from .metrics import evaluate_sample
from .prompting import (
    PromptTemplate,
    build_chain_history,
    build_model_input,
    load_prompt_pack,
    render_prompt,
)
from .prepare import build_prepared_data
from .schema import ChainPairRecord, EvaluationRecord, PreparedSample, RenderedPrompt
from .utils import append_jsonl, ensure_directory, read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class PendingEvaluationTask:
    sample_id: str
    channel: str
    future: Future[EvaluationRecord]
    pair_id: str | None = None
    upstream_sample_id: str | None = None
    downstream_sample_id: str | None = None


def _judge_client_from_config(judge_config: JudgeConfig):
    if judge_config.backend == "static-pass":
        return StaticJudgeClient(always_pass=True)
    if judge_config.backend == "static-fail":
        return StaticJudgeClient(always_pass=False)
    base_url = judge_config.resolved_base_url()
    api_key = judge_config.resolved_api_key()
    if not base_url or not api_key:
        raise ValueError(
            "judge configuration is missing; set [judge].base_url and provide an API key "
            "via [judge].api_key or [judge].api_key_env"
        )
    return OpenAIJudgeClient(
        base_url=base_url,
        api_key=api_key,
        model=judge_config.model,
        temperature=judge_config.temperature,
        top_p=judge_config.top_p,
        top_k=judge_config.top_k,
        max_tokens=judge_config.max_tokens,
        n=judge_config.n,
        seed=judge_config.seed,
        invalid_json_retries=judge_config.invalid_json_retries,
    )


def _load_existing_evaluation_records(path: Path) -> dict[str, EvaluationRecord]:
    if not path.exists():
        return {}
    records: dict[str, EvaluationRecord] = {}
    for row in read_jsonl(path):
        if row.get("task_pass") is None:
            continue
        records[row["sample_id"]] = EvaluationRecord.from_dict(row)
    return records


def _load_existing_prediction_artifacts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return {row["sample_id"]: row.get("raw_output") for row in read_jsonl(path)}


def _load_existing_oracle_pair_results(path: Path) -> dict[str, dict[str, EvaluationRecord]]:
    if not path.exists():
        return {}
    pair_results: dict[str, dict[str, EvaluationRecord]] = {}
    for row in read_jsonl(path):
        upstream = row.get("upstream", {})
        downstream = row.get("downstream", {})
        if upstream.get("task_pass") is None or downstream.get("task_pass") is None:
            continue
        pair_results[row["pair_id"]] = {
            "upstream": EvaluationRecord.from_dict(upstream),
            "downstream": EvaluationRecord.from_dict(downstream),
        }
    return pair_results


def _append_prediction_artifact(path: Path, sample_id: str, raw_output: Any, protocol_id: str) -> None:
    append_jsonl(
        path,
        [
            {
                "sample_id": sample_id,
                "raw_output": raw_output,
                "protocol_id": protocol_id,
            }
        ],
    )


def _append_chain_prediction_artifact(
    path: Path,
    pair: ChainPairRecord,
    raw_output: Any,
    protocol_id: str,
) -> None:
    append_jsonl(
        path,
        [
            {
                "pair_id": pair.pair_id,
                "sample_id": pair.downstream_sample_id,
                "upstream_sample_id": pair.upstream_sample_id,
                "downstream_sample_id": pair.downstream_sample_id,
                "raw_output": raw_output,
                "protocol_id": protocol_id,
            }
        ],
    )


def _append_result(path: Path, record: EvaluationRecord, extra: dict[str, Any] | None = None) -> None:
    payload = record.to_dict()
    if extra:
        payload = {**extra, **payload}
    append_jsonl(path, [payload])


def _append_oracle_pair_result(
    path: Path,
    pair_id: str,
    upstream: EvaluationRecord,
    downstream: EvaluationRecord,
) -> None:
    append_jsonl(
        path,
        [
            {
                "pair_id": pair_id,
                "upstream": upstream.to_dict(),
                "downstream": downstream.to_dict(),
            }
        ],
    )


def _error_summary(
    *,
    sample_id: str,
    stage: str,
    exc: Exception,
    pair_id: str | None = None,
) -> dict[str, str]:
    payload = {
        "sample_id": sample_id,
        "stage": stage,
        "reason": f"{type(exc).__name__}: {exc}",
    }
    if pair_id is not None:
        payload["pair_id"] = pair_id
    return payload


def _default_run_name(model_name: str, protocol_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_name}_{protocol_id}"


def _evaluate_with_judge_client(
    judge_client: Any,
    sample: PreparedSample,
    raw_output: Any,
    *,
    override_tracking_with_gt: bool = False,
) -> EvaluationRecord:
    return evaluate_sample(
        sample,
        raw_output,
        judge_client=judge_client,
        override_tracking_with_gt=override_tracking_with_gt,
    )


def _chain_pairs_by_downstream_id(chain_pairs: list[ChainPairRecord]) -> dict[str, ChainPairRecord]:
    chain_pair_by_downstream_id: dict[str, ChainPairRecord] = {}
    for pair in chain_pairs:
        if pair.downstream_sample_id in chain_pair_by_downstream_id:
            raise ValueError(
                f"multiple chain pairs target the same downstream sample: {pair.downstream_sample_id}"
            )
        chain_pair_by_downstream_id[pair.downstream_sample_id] = pair
    return chain_pair_by_downstream_id


def _load_and_validate_chain_pairs(
    config_chain_manifest: Path | None,
    prepared_by_sample_id: dict[str, PreparedSample],
    target_sample_ids: set[str],
) -> list[ChainPairRecord]:
    spatial_imagination_sample_ids = {
        sample_id
        for sample_id, sample in prepared_by_sample_id.items()
        if sample_id in target_sample_ids and sample.task_name == TASK_SPATIAL_IMAGINATION
    }
    if spatial_imagination_sample_ids and config_chain_manifest is None:
        raise ValueError(
            "run-eval requires [run_eval].chain_manifest when the selected protocol contains "
            "Spatial_Imagination samples"
        )
    if config_chain_manifest is None:
        return []

    chain_pairs = load_chain_pairs(config_chain_manifest)
    chain_pair_by_downstream_id = _chain_pairs_by_downstream_id(chain_pairs)

    missing_downstream = sorted(spatial_imagination_sample_ids - set(chain_pair_by_downstream_id))
    if missing_downstream:
        raise ValueError(
            "chain_manifest is missing Spatial_Imagination samples for this protocol: "
            + ", ".join(missing_downstream)
        )

    for pair in chain_pairs:
        if pair.downstream_sample_id not in target_sample_ids:
            continue
        if pair.upstream_sample_id not in prepared_by_sample_id:
            raise ValueError(
                f"chain pair {pair.pair_id} references upstream sample "
                f"{pair.upstream_sample_id} that is unavailable in prepared data for this protocol"
            )
        if pair.downstream_sample_id not in prepared_by_sample_id:
            raise ValueError(
                f"chain pair {pair.pair_id} references downstream sample "
                f"{pair.downstream_sample_id} that is unavailable in prepared data for this protocol"
            )

    return [pair for pair in chain_pairs if pair.downstream_sample_id in target_sample_ids]


def _render_prompts_for_samples(
    prompt_pack: dict[str, PromptTemplate],
    samples: list[PreparedSample],
) -> dict[str, RenderedPrompt]:
    return {sample.sample_id: render_prompt(prompt_pack, sample) for sample in samples}


def cmd_validate_data(args: argparse.Namespace) -> int:
    config = load_validate_data_config(Path(args.config))
    records, issues = scan_dataset(config.data_root)
    summary = summarize_records(records)
    print(f"Validated {summary['num_samples']} samples across {summary['num_videos']} videos.")
    for task_name, count in summary["task_counts"].items():
        print(f"  {task_name}: {count}")
    if issues:
        print(f"Found {len(issues)} issue(s).")
        for issue in issues[:50]:
            print(f"  - {issue}")
        return 1
    print("No validation issues found.")
    return 0


def cmd_build_chain_manifest(args: argparse.Namespace) -> int:
    config = load_build_chain_manifest_config(Path(args.config))
    pairs = build_chain_manifest(config.data_root, config.out)
    print(f"Wrote {len(pairs)} chain pairs to {config.out}")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    config = load_prepare_data_config(Path(args.config))
    results = build_prepared_data(
        config.data_root,
        config.prepared_root,
        config.protocols,
    )
    for result in results:
        print(
            f"Prepared {result['num_prepared_samples']} samples for {result['protocol_id']} "
            f"at {result['protocol_root']}"
        )
    return 0


def cmd_run_eval(args: argparse.Namespace) -> int:
    config = load_run_eval_config(Path(args.config))
    prepared_samples = load_prepared_for_protocol(config.prepared_root, config.protocol)
    prepared_by_sample_id = {sample.sample_id: sample for sample in prepared_samples}
    prompt_pack = load_prompt_pack(config.prompt_root)

    adapter = resolve_adapter(config.adapter)
    commentary_supported = adapter.supports_commentary()
    model_name = config.model_name or adapter.name

    target_samples = [
        sample
        for sample in prepared_samples
        if not (sample.task_name == "Commentary" and not commentary_supported)
    ]
    target_sample_ids = {sample.sample_id for sample in target_samples}
    rendered_prompts_by_sample_id = _render_prompts_for_samples(prompt_pack, target_samples)

    chain_pairs = _load_and_validate_chain_pairs(
        config.chain_manifest,
        prepared_by_sample_id,
        target_sample_ids,
    )
    chain_pair_by_downstream_id = _chain_pairs_by_downstream_id(chain_pairs)
    chain_downstream_sample_ids = {pair.downstream_sample_id for pair in chain_pairs}
    normal_target_samples = [
        sample for sample in target_samples if sample.sample_id not in chain_downstream_sample_ids
    ]
    normal_target_sample_ids = {sample.sample_id for sample in normal_target_samples}

    artifacts_root = ensure_directory(config.artifacts_root)
    run_dir = ensure_directory(
        artifacts_root / (config.run_name or _default_run_name(model_name, config.protocol))
    )
    predictions_path = run_dir / "predictions.jsonl"
    results_path = run_dir / "results.jsonl"
    chain_predictions_path = run_dir / "chain_predictions.jsonl"
    chain_results_path = run_dir / "chain_results.jsonl"
    summary_path = run_dir / "summary.json"
    oracle_pair_results_path = run_dir / "oracle_pair_results.jsonl"

    normal_results_by_sample_id = _load_existing_evaluation_records(results_path)
    chain_results_by_sample_id = _load_existing_evaluation_records(chain_results_path)
    existing_records_by_sample_id = {**normal_results_by_sample_id, **chain_results_by_sample_id}

    normal_prediction_map = _load_existing_prediction_artifacts(predictions_path)
    for sample_id, record in normal_results_by_sample_id.items():
        if sample_id not in normal_prediction_map:
            normal_prediction_map[sample_id] = record.raw_output

    chain_prediction_map = _load_existing_prediction_artifacts(chain_predictions_path)
    for sample_id, record in chain_results_by_sample_id.items():
        if sample_id not in chain_prediction_map:
            chain_prediction_map[sample_id] = record.raw_output

    completed_before_run = len(
        [sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids]
    )
    if completed_before_run:
        print(f"Resuming run from {run_dir} with {completed_before_run} completed sample(s).")

    judge_client = _judge_client_from_config(config.judge)

    prediction_errors_this_run: list[dict[str, str]] = []
    evaluation_errors_this_run: list[dict[str, str]] = []
    chain_prediction_errors_this_run: list[dict[str, str]] = []
    chain_evaluation_errors_this_run: list[dict[str, str]] = []
    blocked_chain_sample_ids: list[str] = []
    completed_normal_sample_ids_this_run: list[str] = []
    completed_chain_sample_ids_this_run: list[str] = []

    def submit_evaluation(
        executor: ThreadPoolExecutor,
        *,
        sample: PreparedSample,
        raw_output: Any,
        channel: str,
        pair: ChainPairRecord | None = None,
    ) -> PendingEvaluationTask:
        future = executor.submit(
            _evaluate_with_judge_client,
            judge_client,
            sample,
            raw_output,
        )
        return PendingEvaluationTask(
            sample_id=sample.sample_id,
            channel=channel,
            future=future,
            pair_id=pair.pair_id if pair else None,
            upstream_sample_id=pair.upstream_sample_id if pair else None,
            downstream_sample_id=pair.downstream_sample_id if pair else None,
        )

    def handle_completed_task(task: PendingEvaluationTask) -> None:
        try:
            record = task.future.result()
        except Exception as exc:  # noqa: BLE001
            target_errors = (
                chain_evaluation_errors_this_run
                if task.channel == "chain"
                else evaluation_errors_this_run
            )
            target_errors.append(
                _error_summary(
                    sample_id=task.sample_id,
                    stage=(
                        "judge"
                        if isinstance(exc, JudgeResponseFormatExhaustedError)
                        else "evaluation"
                    ),
                    exc=exc,
                    pair_id=task.pair_id,
                )
            )
            return

        existing_records_by_sample_id[task.sample_id] = record
        if task.channel == "chain":
            chain_results_by_sample_id[task.sample_id] = record
            completed_chain_sample_ids_this_run.append(task.sample_id)
            _append_result(
                chain_results_path,
                record,
                extra={
                    "pair_id": task.pair_id,
                    "upstream_sample_id": task.upstream_sample_id,
                    "downstream_sample_id": task.downstream_sample_id,
                },
            )
        else:
            normal_results_by_sample_id[task.sample_id] = record
            completed_normal_sample_ids_this_run.append(task.sample_id)
            _append_result(results_path, record)

    def drain_completed_tasks(pending_tasks: dict[str, PendingEvaluationTask], *, wait_all: bool) -> None:
        if wait_all:
            sample_ids = list(pending_tasks)
        else:
            sample_ids = [
                sample_id
                for sample_id, task in pending_tasks.items()
                if task.future.done()
            ]
        for sample_id in sample_ids:
            task = pending_tasks.pop(sample_id)
            handle_completed_task(task)

    with ThreadPoolExecutor(
        max_workers=config.judge.concurrency,
        thread_name_prefix="omnichain-judge",
    ) as executor:
        pending_tasks: dict[str, PendingEvaluationTask] = {}

        for sample in normal_target_samples:
            if sample.sample_id in existing_records_by_sample_id:
                drain_completed_tasks(pending_tasks, wait_all=False)
                continue
            raw_output = normal_prediction_map.get(sample.sample_id)
            if raw_output is None:
                try:
                    raw_output = adapter.predict(
                        build_model_input(
                            sample,
                            rendered_prompts_by_sample_id[sample.sample_id],
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    prediction_errors_this_run.append(
                        _error_summary(sample_id=sample.sample_id, stage="prediction", exc=exc)
                    )
                    drain_completed_tasks(pending_tasks, wait_all=False)
                    continue
                normal_prediction_map[sample.sample_id] = raw_output
                _append_prediction_artifact(
                    predictions_path,
                    sample.sample_id,
                    raw_output,
                    config.protocol,
                )
            if sample.sample_id not in pending_tasks:
                pending_tasks[sample.sample_id] = submit_evaluation(
                    executor,
                    sample=sample,
                    raw_output=raw_output,
                    channel="normal",
                )
            drain_completed_tasks(pending_tasks, wait_all=False)

        drain_completed_tasks(pending_tasks, wait_all=False)

        for pair in chain_pairs:
            downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
            if downstream_sample.sample_id in existing_records_by_sample_id:
                drain_completed_tasks(pending_tasks, wait_all=False)
                continue

            upstream_raw_output = normal_prediction_map.get(pair.upstream_sample_id)
            if upstream_raw_output is None:
                blocked_chain_sample_ids.append(pair.downstream_sample_id)
                drain_completed_tasks(pending_tasks, wait_all=False)
                continue

            raw_output = chain_prediction_map.get(pair.downstream_sample_id)
            if raw_output is None:
                try:
                    raw_output = adapter.predict(
                        build_model_input(
                            downstream_sample,
                            rendered_prompts_by_sample_id[downstream_sample.sample_id],
                            conversation_history=build_chain_history(
                                prepared_by_sample_id[pair.upstream_sample_id],
                                upstream_raw_output,
                            ),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    chain_prediction_errors_this_run.append(
                        _error_summary(
                            sample_id=pair.downstream_sample_id,
                            stage="chain_prediction",
                            exc=exc,
                            pair_id=pair.pair_id,
                        )
                    )
                    drain_completed_tasks(pending_tasks, wait_all=False)
                    continue
                chain_prediction_map[pair.downstream_sample_id] = raw_output
                _append_chain_prediction_artifact(
                    chain_predictions_path,
                    pair,
                    raw_output,
                    config.protocol,
                )

            if pair.downstream_sample_id not in pending_tasks:
                pending_tasks[pair.downstream_sample_id] = submit_evaluation(
                    executor,
                    sample=downstream_sample,
                    raw_output=raw_output,
                    channel="chain",
                    pair=pair,
                )
            drain_completed_tasks(pending_tasks, wait_all=False)

        drain_completed_tasks(pending_tasks, wait_all=True)

    ordered_records = [
        existing_records_by_sample_id[sample.sample_id]
        for sample in prepared_samples
        if sample.sample_id in existing_records_by_sample_id
    ]
    evaluation = summarize_evaluation_records(
        prepared_samples,
        ordered_records,
        model_name=model_name,
        commentary_supported=commentary_supported,
        enable_bertscore=config.enable_bertscore,
    )

    oracle_summary = None
    oracle_pair_results = None
    oracle_errors_this_run: list[dict[str, str]] = []
    if config.chain_manifest:
        oracle_pair_results = _load_existing_oracle_pair_results(oracle_pair_results_path)
        if config.enable_oracle_track:
            if not adapter.supports_oracle_track():
                raise ValueError(f"adapter {adapter.name} does not support oracle_track reruns")
            try:
                fresh_oracle_pair_results = evaluate_oracle_chain(
                    adapter,
                    prepared_by_sample_id,
                    chain_pairs,
                    prompt_pack=prompt_pack,
                    judge_client=judge_client,
                )
            except Exception as exc:  # noqa: BLE001
                oracle_errors_this_run.append(
                    {
                        "pair_id": "batch",
                        "stage": (
                            "oracle_judge"
                            if isinstance(exc, JudgeResponseFormatExhaustedError)
                            else "oracle_evaluation"
                        ),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                for pair in chain_pairs:
                    pair_result = fresh_oracle_pair_results.get(pair.pair_id)
                    if pair_result is None:
                        continue
                    oracle_pair_results[pair.pair_id] = pair_result
                    _append_oracle_pair_result(
                        oracle_pair_results_path,
                        pair.pair_id,
                        pair_result["upstream"],
                        pair_result["downstream"],
                    )

        oracle_summary = summarize_experiment_b(
            chain_pairs,
            evaluation["records_by_sample_id"],
            oracle_pair_results=oracle_pair_results,
        )

    write_jsonl(
        predictions_path,
        [
            {"sample_id": sample_id, "raw_output": raw_output, "protocol_id": config.protocol}
            for sample_id, raw_output in sorted(normal_prediction_map.items())
        ],
    )
    write_jsonl(
        results_path,
        [record.to_dict() for sample_id, record in sorted(normal_results_by_sample_id.items())],
    )
    write_jsonl(
        chain_predictions_path,
        [
            {
                "pair_id": chain_pair_by_downstream_id[sample_id].pair_id,
                "sample_id": sample_id,
                "upstream_sample_id": chain_pair_by_downstream_id[sample_id].upstream_sample_id,
                "downstream_sample_id": chain_pair_by_downstream_id[sample_id].downstream_sample_id,
                "raw_output": raw_output,
                "protocol_id": config.protocol,
            }
            for sample_id, raw_output in sorted(chain_prediction_map.items())
            if sample_id in chain_pair_by_downstream_id
        ],
    )
    write_jsonl(
        chain_results_path,
        [
            {
                "pair_id": chain_pair_by_downstream_id[sample_id].pair_id,
                "upstream_sample_id": chain_pair_by_downstream_id[sample_id].upstream_sample_id,
                "downstream_sample_id": chain_pair_by_downstream_id[sample_id].downstream_sample_id,
                **record.to_dict(),
            }
            for sample_id, record in sorted(chain_results_by_sample_id.items())
            if sample_id in chain_pair_by_downstream_id
        ],
    )
    if oracle_pair_results is not None:
        write_jsonl(
            oracle_pair_results_path,
            [
                {
                    "pair_id": pair_id,
                    "upstream": pair_result["upstream"].to_dict(),
                    "downstream": pair_result["downstream"].to_dict(),
                }
                for pair_id, pair_result in sorted(oracle_pair_results.items())
            ],
        )

    completed_total = len(
        [sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids]
    )
    normal_predicted_sample_ids = normal_target_sample_ids & set(normal_prediction_map)
    normal_completed_sample_ids = normal_target_sample_ids & set(normal_results_by_sample_id)
    chain_target_sample_ids = chain_downstream_sample_ids & target_sample_ids
    chain_predicted_sample_ids = chain_target_sample_ids & set(chain_prediction_map)
    chain_completed_sample_ids = chain_target_sample_ids & set(chain_results_by_sample_id)

    run_status = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "run_name": run_dir.name,
        "total_target_samples": len(target_samples),
        "completed_samples_before_run": completed_before_run,
        "completed_normal_samples_this_run": len(completed_normal_sample_ids_this_run),
        "completed_chain_samples_this_run": len(completed_chain_sample_ids_this_run),
        "completed_samples_total": completed_total,
        "pending_prediction_sample_ids": sorted(normal_target_sample_ids - normal_predicted_sample_ids),
        "predicted_not_evaluated_sample_ids": sorted(
            normal_predicted_sample_ids - normal_completed_sample_ids
        ),
        "pending_chain_prediction_sample_ids": sorted(
            chain_target_sample_ids - chain_predicted_sample_ids
        ),
        "chain_predicted_not_evaluated_sample_ids": sorted(
            chain_predicted_sample_ids - chain_completed_sample_ids
        ),
        "blocked_chain_sample_ids": sorted(set(blocked_chain_sample_ids)),
        "prediction_errors_this_run": prediction_errors_this_run,
        "evaluation_errors_this_run": evaluation_errors_this_run,
        "chain_prediction_errors_this_run": chain_prediction_errors_this_run,
        "chain_evaluation_errors_this_run": chain_evaluation_errors_this_run,
        "oracle_errors_this_run": oracle_errors_this_run,
    }
    summary_payload = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "overall": evaluation["overall"],
        "task_summaries": [summary.to_dict() for summary in evaluation["task_summaries"]],
        "experiment_b": oracle_summary,
        "commentary_supported": commentary_supported,
        "run_status": run_status,
    }
    write_json(summary_path, summary_payload)
    print(
        "Run status: "
        f"{completed_total}/{len(target_samples)} completed, "
        f"{len(run_status['pending_prediction_sample_ids'])} pending prediction, "
        f"{len(run_status['pending_chain_prediction_sample_ids'])} pending chain prediction."
    )
    print(f"Wrote evaluation artifacts to {run_dir}")
    return 0


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to the TOML config file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnichain-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data")
    _add_config_argument(validate_parser)
    validate_parser.set_defaults(func=cmd_validate_data)

    chain_parser = subparsers.add_parser("build-chain-manifest")
    _add_config_argument(chain_parser)
    chain_parser.set_defaults(func=cmd_build_chain_manifest)

    prepare_parser = subparsers.add_parser("prepare-data")
    _add_config_argument(prepare_parser)
    prepare_parser.set_defaults(func=cmd_prepare_data)

    run_parser = subparsers.add_parser("run-eval")
    _add_config_argument(run_parser)
    run_parser.set_defaults(func=cmd_run_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
