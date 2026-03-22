"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .adapters import resolve_adapter
from .config import (
    JudgeConfig,
    load_build_chain_manifest_config,
    load_prepare_data_config,
    load_report_config,
    load_run_eval_config,
    load_validate_data_config,
)
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
from .prepare import build_prepared_data
from .schema import EvaluationRecord
from .utils import append_jsonl, ensure_directory, read_jsonl, write_json, write_jsonl


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


def _load_prediction_map(path: Path) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for row in read_jsonl(path):
        payload = row.get("normalized_prediction", row.get("raw_output"))
        mapping[row["sample_id"]] = payload
    return mapping


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


def _append_sample_result(path: Path, record: EvaluationRecord) -> None:
    append_jsonl(path, [record.to_dict()])


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
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "stage": stage,
        "reason": f"{type(exc).__name__}: {exc}",
    }


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
        if result["dataset_issue_count"]:
            print(
                f"  warnings: skipped {result['dataset_issue_count']} invalid annotation issue(s); "
                "see build_manifest.json for details"
            )
    return 0


def _default_run_name(model_name: str, protocol_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_name}_{protocol_id}"


def cmd_run_eval(args: argparse.Namespace) -> int:
    config = load_run_eval_config(Path(args.config))
    prepared_root = config.prepared_root
    prepared_samples = load_prepared_for_protocol(prepared_root, config.protocol)
    prepared_by_sample_id = {sample.sample_id: sample for sample in prepared_samples}

    prediction_source_map: dict[str, Any] | None = None
    commentary_supported = not config.commentary_unsupported
    adapter = None
    if config.adapter:
        adapter = resolve_adapter(config.adapter)
        commentary_supported = adapter.supports_commentary()
        model_name = config.model_name or adapter.name
    else:
        prediction_source_map = _load_prediction_map(config.predictions)
        model_name = config.model_name or config.predictions.stem

    target_samples = [
        sample
        for sample in prepared_samples
        if not (sample.task_name == "Commentary" and not commentary_supported)
    ]
    target_sample_ids = {sample.sample_id for sample in target_samples}
    artifacts_root = ensure_directory(config.artifacts_root)
    run_dir = ensure_directory(
        artifacts_root / (config.run_name or _default_run_name(model_name, config.protocol))
    )
    predictions_artifact_path = run_dir / "predictions.jsonl"
    sample_results_path = run_dir / "sample_results.jsonl"
    oracle_pair_results_path = run_dir / "oracle_pair_results.jsonl"
    existing_records_by_sample_id = _load_existing_evaluation_records(sample_results_path)
    artifact_prediction_map = _load_existing_prediction_artifacts(predictions_artifact_path)
    for sample_id, record in existing_records_by_sample_id.items():
        if sample_id not in artifact_prediction_map:
            artifact_prediction_map[sample_id] = record.raw_output
    completed_before_run = len([sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids])
    if completed_before_run:
        print(f"Resuming run from {run_dir} with {completed_before_run} completed sample(s).")

    judge_client = _judge_client_from_config(config.judge)
    from .metrics import evaluate_sample

    prediction_errors_this_run: list[dict[str, str]] = []
    evaluation_errors_this_run: list[dict[str, str]] = []
    completed_sample_ids_this_run: list[str] = []
    for sample in target_samples:
        if sample.sample_id in existing_records_by_sample_id:
            continue
        if sample.sample_id in artifact_prediction_map:
            raw_output = artifact_prediction_map[sample.sample_id]
        else:
            try:
                if adapter is not None:
                    raw_output = adapter.predict(sample)
                else:
                    raw_output = (
                        prediction_source_map.get(sample.sample_id)
                        if prediction_source_map
                        else None
                    )
            except Exception as exc:  # noqa: BLE001
                prediction_errors_this_run.append(
                    _error_summary(sample_id=sample.sample_id, stage="prediction", exc=exc)
                )
                continue
            artifact_prediction_map[sample.sample_id] = raw_output
            _append_prediction_artifact(
                predictions_artifact_path,
                sample.sample_id,
                raw_output,
                config.protocol,
            )
        try:
            record = evaluate_sample(
                sample,
                raw_output,
                judge_client=judge_client,
            )
        except Exception as exc:  # noqa: BLE001
            evaluation_errors_this_run.append(
                _error_summary(
                    sample_id=sample.sample_id,
                    stage="judge" if isinstance(exc, JudgeResponseFormatExhaustedError) else "evaluation",
                    exc=exc,
                )
            )
            continue
        existing_records_by_sample_id[sample.sample_id] = record
        _append_sample_result(sample_results_path, record)
        completed_sample_ids_this_run.append(sample.sample_id)

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
        chain_pairs = load_chain_pairs(config.chain_manifest)
        oracle_pair_results = _load_existing_oracle_pair_results(oracle_pair_results_path)
        if adapter is not None and config.enable_oracle_track:
            if not adapter.supports_oracle_track():
                raise ValueError(f"adapter {adapter.name} does not support oracle_track reruns")
            for pair in chain_pairs:
                if pair.pair_id in oracle_pair_results:
                    continue
                try:
                    pair_result = evaluate_oracle_chain(
                        adapter,
                        prepared_by_sample_id,
                        [pair],
                        judge_client=judge_client,
                    )[pair.pair_id]
                except Exception as exc:  # noqa: BLE001
                    oracle_errors_this_run.append(
                        {
                            "pair_id": pair.pair_id,
                            "stage": (
                                "oracle_judge"
                                if isinstance(exc, JudgeResponseFormatExhaustedError)
                                else "oracle_evaluation"
                            ),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                oracle_pair_results[pair.pair_id] = pair_result
                _append_oracle_pair_result(
                    oracle_pair_results_path,
                    pair.pair_id,
                    pair_result["upstream"],
                    pair_result["downstream"],
                )
        elif config.oracle_predictions:
            oracle_prediction_map = _load_prediction_map(config.oracle_predictions)
            for pair in chain_pairs:
                if pair.pair_id in oracle_pair_results:
                    continue
                upstream_sample = prepared_by_sample_id[pair.upstream_sample_id]
                downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
                try:
                    pair_result = {
                        "upstream": evaluate_sample(
                            upstream_sample,
                            oracle_prediction_map.get(upstream_sample.sample_id),
                            judge_client=judge_client,
                            override_tracking_with_gt=upstream_sample.task_name
                            in {"Continuous_Actions_Caption", "Spatial_Temporal_Grounding"},
                        ),
                        "downstream": evaluate_sample(
                            downstream_sample,
                            oracle_prediction_map.get(downstream_sample.sample_id),
                            judge_client=judge_client,
                        ),
                    }
                except Exception as exc:  # noqa: BLE001
                    oracle_errors_this_run.append(
                        {
                            "pair_id": pair.pair_id,
                            "stage": (
                                "oracle_judge"
                                if isinstance(exc, JudgeResponseFormatExhaustedError)
                                else "oracle_evaluation"
                            ),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
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

    prediction_rows = [
        {"sample_id": sample_id, "raw_output": raw_output, "protocol_id": config.protocol}
        for sample_id, raw_output in sorted(artifact_prediction_map.items())
    ]
    write_jsonl(run_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(run_dir / "sample_results.jsonl", [record.to_dict() for record in evaluation["records"]])
    if oracle_pair_results is not None:
        write_jsonl(
            run_dir / "oracle_pair_results.jsonl",
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
    predicted_sample_ids = {
        sample_id for sample_id in artifact_prediction_map if sample_id in target_sample_ids
    }
    completed_sample_ids = {
        sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids
    }
    pending_prediction_sample_ids = sorted(target_sample_ids - predicted_sample_ids)
    predicted_not_evaluated_sample_ids = sorted(predicted_sample_ids - completed_sample_ids)
    run_status = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "run_name": run_dir.name,
        "total_target_samples": len(target_samples),
        "completed_samples_before_run": completed_before_run,
        "completed_samples_this_run": len(completed_sample_ids_this_run),
        "completed_samples_total": completed_total,
        "pending_prediction_samples_total": len(pending_prediction_sample_ids),
        "predicted_not_evaluated_samples_total": len(predicted_not_evaluated_sample_ids),
        "pending_prediction_sample_ids": pending_prediction_sample_ids,
        "predicted_not_evaluated_sample_ids": predicted_not_evaluated_sample_ids,
        "prediction_errors_this_run": prediction_errors_this_run,
        "evaluation_errors_this_run": evaluation_errors_this_run,
        "oracle_errors_this_run": oracle_errors_this_run,
    }
    write_json(run_dir / "run_status.json", run_status)

    write_json(run_dir / "task_summaries.json", [summary.to_dict() for summary in evaluation["task_summaries"]])
    summary_payload = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "overall": evaluation["overall"],
        "task_summaries": [summary.to_dict() for summary in evaluation["task_summaries"]],
        "experiment_b": oracle_summary,
        "commentary_supported": commentary_supported,
        "run_status": run_status,
    }
    write_json(run_dir / "summary.json", summary_payload)
    print(
        "Run status: "
        f"{completed_total}/{len(target_samples)} completed, "
        f"{len(pending_prediction_sample_ids)} pending prediction, "
        f"{len(predicted_not_evaluated_sample_ids)} predicted-not-evaluated."
    )
    print(f"Wrote evaluation artifacts to {run_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = load_report_config(Path(args.config))
    artifacts_root = config.artifacts_root
    summary_paths = sorted(artifacts_root.rglob("summary.json"))
    summaries = []
    for summary_path in summary_paths:
        summaries.append({"path": str(summary_path), **json.loads(summary_path.read_text())})
    output_path = config.out if config.out else artifacts_root / "report.json"
    write_json(output_path, {"runs": summaries})
    print(f"Wrote consolidated report with {len(summaries)} run(s) to {output_path}")
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

    report_parser = subparsers.add_parser("report")
    _add_config_argument(report_parser)
    report_parser.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
