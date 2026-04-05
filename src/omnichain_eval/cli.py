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
    StructurerConfig,
    load_build_chain_manifest_config,
    load_prepare_data_config,
    load_run_eval_config,
    load_validate_data_config,
)
from .constants import ORACLE_EXPERIMENT_B_VARIANTS, TASK_SPATIAL_IMAGINATION
from .dataset import scan_dataset_report, summarize_scan_report
from .experiments import (
    OraclePairError,
    build_chain_manifest,
    evaluate_oracle_chain_pair,
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
    load_oracle_prompt_pack,
    load_prompt_pack,
    render_prompt,
)
from .prepare import build_prepared_data
from .schema import (
    ChainPairRecord,
    EvaluationRecord,
    PreparedSample,
    RenderedPrompt,
    StructuredPredictionRecord,
)
from .structurer import (
    OpenAIStructurerBackend,
    StaticParseStructurerBackend,
    StructurerResponseFormatExhaustedError,
    StructurerService,
    load_oracle_structurer_prompt_pack,
    load_structurer_prompt_pack,
)
from .utils import append_jsonl, ensure_directory, read_json, read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class PendingStructuringTask:
    sample: PreparedSample
    channel: str
    future: Future[StructuredPredictionRecord]
    pair_id: str | None = None
    upstream_sample_id: str | None = None
    downstream_sample_id: str | None = None


@dataclass(slots=True)
class PendingEvaluationTask:
    sample: PreparedSample
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
        prompt_root=judge_config.prompt_root,
        model=judge_config.model,
        temperature=judge_config.temperature,
        extra_body=judge_config.extra_body,
        invalid_json_retries=judge_config.invalid_json_retries,
    )


def _structurer_service_from_config(structurer_config: StructurerConfig) -> StructurerService:
    prompt_pack = load_structurer_prompt_pack(structurer_config.prompt_root)
    oracle_prompt_pack = (
        load_oracle_structurer_prompt_pack(structurer_config.oracle_prompt_root)
        if structurer_config.oracle_prompt_root is not None
        else None
    )
    if structurer_config.backend == "static-parse":
        backend = StaticParseStructurerBackend()
    else:
        base_url = structurer_config.resolved_base_url()
        api_key = structurer_config.resolved_api_key()
        if not base_url or not api_key:
            raise ValueError(
                "structurer configuration is missing; set [structurer].base_url and provide "
                "an API key via [structurer].api_key or [structurer].api_key_env"
            )
        backend = OpenAIStructurerBackend(
            base_url=base_url,
            api_key=api_key,
            model=structurer_config.model,
            temperature=structurer_config.temperature,
            extra_body=structurer_config.extra_body,
        )
    return StructurerService(
        backend=backend,
        prompt_pack=prompt_pack,
        oracle_prompt_pack=oracle_prompt_pack,
        invalid_json_retries=structurer_config.invalid_json_retries,
    )


def _load_existing_evaluation_records(path: Path) -> dict[str, EvaluationRecord]:
    if not path.exists():
        return {}
    records: dict[str, EvaluationRecord] = {}
    for row in read_jsonl(path):
        if row.get("task_pass") is None:
            continue
        record = EvaluationRecord.from_dict(row)
        records[record.sample_id] = record
    return records


def _load_existing_prediction_artifacts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows: dict[str, str] = {}
    for row in read_jsonl(path):
        raw_output = row.get("raw_output")
        if raw_output is None:
            continue
        rows[row["sample_id"]] = str(raw_output)
    return rows


def _load_existing_structured_records(path: Path) -> dict[str, StructuredPredictionRecord]:
    if not path.exists():
        return {}
    records: dict[str, StructuredPredictionRecord] = {}
    for row in read_jsonl(path):
        if row.get("structured_prediction") is None:
            continue
        record = StructuredPredictionRecord.from_dict(row)
        records[record.sample_id] = record
    return records


def _structured_record_from_evaluation(record: EvaluationRecord) -> StructuredPredictionRecord | None:
    if record.structured_prediction is None or record.raw_output is None:
        return None
    return StructuredPredictionRecord(
        sample_id=record.sample_id,
        task_name=record.task_name,
        video_key=record.video_key,
        protocol_id=record.protocol_id,
        raw_output=record.raw_output,
        structured_prediction=record.structured_prediction,
        structuring_errors=list(record.structuring_errors),
        structuring_warnings=list(record.structuring_warnings),
    )


def _backfill_prediction_and_structuring_state(
    *,
    prediction_map: dict[str, str],
    structured_map: dict[str, StructuredPredictionRecord],
    result_map: dict[str, EvaluationRecord],
) -> None:
    for sample_id, structured_record in structured_map.items():
        prediction_map.setdefault(sample_id, structured_record.raw_output)
    for sample_id, result_record in result_map.items():
        if result_record.raw_output is not None:
            prediction_map.setdefault(sample_id, result_record.raw_output)
        if sample_id in structured_map:
            continue
        structured_record = _structured_record_from_evaluation(result_record)
        if structured_record is not None:
            structured_map[sample_id] = structured_record


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


def _append_prediction_artifact(path: Path, sample_id: str, raw_output: str, protocol_id: str) -> None:
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
    raw_output: str,
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


def _append_structured_artifact(path: Path, record: StructuredPredictionRecord) -> None:
    append_jsonl(path, [record.to_dict()])


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


def _structure_sample(
    structurer_service: StructurerService,
    sample: PreparedSample,
    raw_output: str,
    *,
    pair: ChainPairRecord | None = None,
) -> StructuredPredictionRecord:
    result = structurer_service.structure(sample, raw_output)
    return StructuredPredictionRecord(
        sample_id=sample.sample_id,
        task_name=sample.task_name,
        video_key=sample.video_key,
        protocol_id=sample.protocol_id,
        raw_output=result.raw_output,
        structured_prediction=result.structured_prediction,
        structuring_errors=list(result.errors),
        structuring_warnings=list(result.warnings),
        structurer_raw_response=result.structurer_raw_response,
        pair_id=pair.pair_id if pair else None,
        upstream_sample_id=pair.upstream_sample_id if pair else None,
        downstream_sample_id=pair.downstream_sample_id if pair else None,
    )


def _evaluate_with_judge_client(
    judge_client: Any,
    sample: PreparedSample,
    structured_record: StructuredPredictionRecord,
) -> EvaluationRecord:
    return evaluate_sample(
        sample,
        structured_record,
        judge_client=judge_client,
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
    scan_report = scan_dataset_report(config.data_root)
    status = summarize_scan_report(scan_report)
    raw_summary = status["raw_dataset_summary"]
    supported_summary = status["supported_dataset_summary"]
    print(
        f"Validated {raw_summary['num_samples']} raw samples across {raw_summary['num_videos']} videos."
    )
    print(
        f"Supported samples: {supported_summary['num_samples']} "
        f"across {supported_summary['num_videos']} videos."
    )
    if status["ignored_unsupported_sample_count"]:
        print(
            f"Ignored {status['ignored_unsupported_sample_count']} unsupported sample(s)."
        )
        for task_name, count in status["ignored_unsupported_task_counts"].items():
            print(f"  unsupported {task_name}: {count}")
    for task_name, count in supported_summary["task_counts"].items():
        print(f"  supported {task_name}: {count}")
    if scan_report.issues:
        print(f"Found {len(scan_report.issues)} supported issue(s).")
        for issue in scan_report.issues[:50]:
            print(f"  - {issue}")
        return 1
    print("No supported-task validation issues found.")
    return 0


def cmd_build_chain_manifest(args: argparse.Namespace) -> int:
    config = load_build_chain_manifest_config(Path(args.config))
    scan_report = scan_dataset_report(config.data_root)
    pairs = build_chain_manifest(config.data_root, config.out, scan_report=scan_report)
    print(f"Wrote {len(pairs)} chain pairs to {config.out}")
    if scan_report.unsupported_samples:
        print(
            f"Ignored {len(scan_report.unsupported_samples)} unsupported sample(s) while scanning."
        )
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    config = load_prepare_data_config(Path(args.config))
    results = build_prepared_data(
        config.data_root,
        config.prepared_root,
        config.protocols,
        media_formats=config.media_formats,
        generate_oracle_visual_media=config.generate_oracle_visual_media,
        workers=config.workers,
    )
    for result in results:
        print(
            f"Prepared {result['num_prepared_samples']} samples for {result['protocol_id']} "
            f"at {result['protocol_root']}"
        )
        if result["ignored_unsupported_sample_count"]:
            print(
                f"  ignored unsupported samples: {result['ignored_unsupported_sample_count']}"
            )
    return 0


def cmd_run_eval(args: argparse.Namespace) -> int:
    config = load_run_eval_config(Path(args.config))
    prepared_samples = load_prepared_for_protocol(config.prepared_root, config.protocol)
    prepared_by_sample_id = {sample.sample_id: sample for sample in prepared_samples}
    prompt_pack = load_prompt_pack(config.prompt_root)
    oracle_prompt_pack = (
        load_oracle_prompt_pack(config.oracle_prompt_root)
        if config.oracle_prompt_root is not None
        else None
    )
    structurer_service = _structurer_service_from_config(config.structurer)

    adapter = resolve_adapter(config.adapter)
    model_name = config.model_name or adapter.name

    target_samples = list(prepared_samples)
    target_sample_ids = {sample.sample_id for sample in target_samples}
    rendered_prompts_by_sample_id = _render_prompts_for_samples(prompt_pack, target_samples)

    chain_pairs = _load_and_validate_chain_pairs(
        config.chain_manifest,
        prepared_by_sample_id,
        target_sample_ids,
    )
    if config.enable_oracle_track:
        missing_oracle_visual_media = sorted(
            {
                pair.upstream_sample_id
                for pair in chain_pairs
                if not prepared_by_sample_id[pair.upstream_sample_id].oracle_visual_frame_files
                or prepared_by_sample_id[pair.upstream_sample_id].oracle_visual_sampled_video_file
                is None
            }
        )
        if missing_oracle_visual_media:
            raise ValueError(
                "OracleTrack visual variants require prepared data with oracle visual media; "
                "rebuild prepared data with [prepare_data].generate_oracle_visual_media = true. "
                f"Missing upstream sample(s): {', '.join(missing_oracle_visual_media[:5])}"
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
    structured_predictions_path = run_dir / "structured_predictions.jsonl"
    results_path = run_dir / "results.jsonl"
    chain_predictions_path = run_dir / "chain_predictions.jsonl"
    chain_structured_predictions_path = run_dir / "chain_structured_predictions.jsonl"
    chain_results_path = run_dir / "chain_results.jsonl"
    summary_path = run_dir / "summary.json"
    oracle_pair_results_paths = {
        variant: run_dir / f"oracle_{variant}_pair_results.jsonl"
        for variant in ORACLE_EXPERIMENT_B_VARIANTS
    }

    normal_results_by_sample_id = _load_existing_evaluation_records(results_path)
    chain_results_by_sample_id = _load_existing_evaluation_records(chain_results_path)
    normal_structured_by_sample_id = _load_existing_structured_records(structured_predictions_path)
    chain_structured_by_sample_id = _load_existing_structured_records(
        chain_structured_predictions_path
    )
    normal_prediction_map = _load_existing_prediction_artifacts(predictions_path)
    chain_prediction_map = _load_existing_prediction_artifacts(chain_predictions_path)

    _backfill_prediction_and_structuring_state(
        prediction_map=normal_prediction_map,
        structured_map=normal_structured_by_sample_id,
        result_map=normal_results_by_sample_id,
    )
    _backfill_prediction_and_structuring_state(
        prediction_map=chain_prediction_map,
        structured_map=chain_structured_by_sample_id,
        result_map=chain_results_by_sample_id,
    )

    existing_records_by_sample_id = {**normal_results_by_sample_id, **chain_results_by_sample_id}
    completed_before_run = len(
        [sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids]
    )
    if completed_before_run:
        print(f"Resuming run from {run_dir} with {completed_before_run} completed sample(s).")

    judge_client = _judge_client_from_config(config.judge)

    prediction_errors_this_run: list[dict[str, str]] = []
    structuring_errors_this_run: list[dict[str, str]] = []
    evaluation_errors_this_run: list[dict[str, str]] = []
    chain_prediction_errors_this_run: list[dict[str, str]] = []
    chain_structuring_errors_this_run: list[dict[str, str]] = []
    chain_evaluation_errors_this_run: list[dict[str, str]] = []
    blocked_chain_sample_ids: list[str] = []
    structured_normal_sample_ids_this_run: list[str] = []
    structured_chain_sample_ids_this_run: list[str] = []
    completed_normal_sample_ids_this_run: list[str] = []
    completed_chain_sample_ids_this_run: list[str] = []

    with ThreadPoolExecutor(
        max_workers=config.structurer.concurrency,
        thread_name_prefix="omnichain-structurer",
    ) as structurer_executor, ThreadPoolExecutor(
        max_workers=config.judge.concurrency,
        thread_name_prefix="omnichain-judge",
    ) as judge_executor:
        pending_structuring_tasks: dict[str, PendingStructuringTask] = {}
        pending_evaluation_tasks: dict[str, PendingEvaluationTask] = {}

        def submit_structuring(
            *,
            sample: PreparedSample,
            raw_output: str,
            channel: str,
            pair: ChainPairRecord | None = None,
        ) -> PendingStructuringTask:
            future = structurer_executor.submit(
                _structure_sample,
                structurer_service,
                sample,
                raw_output,
                pair=pair,
            )
            return PendingStructuringTask(
                sample=sample,
                channel=channel,
                future=future,
                pair_id=pair.pair_id if pair else None,
                upstream_sample_id=pair.upstream_sample_id if pair else None,
                downstream_sample_id=pair.downstream_sample_id if pair else None,
            )

        def submit_evaluation(
            *,
            sample: PreparedSample,
            structured_record: StructuredPredictionRecord,
            channel: str,
            pair: ChainPairRecord | None = None,
        ) -> PendingEvaluationTask:
            future = judge_executor.submit(
                _evaluate_with_judge_client,
                judge_client,
                sample,
                structured_record,
            )
            return PendingEvaluationTask(
                sample=sample,
                channel=channel,
                future=future,
                pair_id=pair.pair_id if pair else None,
                upstream_sample_id=pair.upstream_sample_id if pair else None,
                downstream_sample_id=pair.downstream_sample_id if pair else None,
            )

        def handle_completed_structuring_task(task: PendingStructuringTask) -> None:
            try:
                record = task.future.result()
            except Exception as exc:  # noqa: BLE001
                target_errors = (
                    chain_structuring_errors_this_run
                    if task.channel == "chain"
                    else structuring_errors_this_run
                )
                target_errors.append(
                    _error_summary(
                        sample_id=task.sample.sample_id,
                        stage=(
                            "structuring"
                            if isinstance(exc, StructurerResponseFormatExhaustedError)
                            else "structuring"
                        ),
                        exc=exc,
                        pair_id=task.pair_id,
                    )
                )
                return

            if task.channel == "chain":
                chain_structured_by_sample_id[task.sample.sample_id] = record
                structured_chain_sample_ids_this_run.append(task.sample.sample_id)
                _append_structured_artifact(chain_structured_predictions_path, record)
            else:
                normal_structured_by_sample_id[task.sample.sample_id] = record
                structured_normal_sample_ids_this_run.append(task.sample.sample_id)
                _append_structured_artifact(structured_predictions_path, record)

            if task.sample.sample_id in existing_records_by_sample_id:
                return
            if task.sample.sample_id in pending_evaluation_tasks:
                return
            pending_evaluation_tasks[task.sample.sample_id] = submit_evaluation(
                sample=task.sample,
                structured_record=record,
                channel=task.channel,
                pair=(
                    chain_pair_by_downstream_id[task.sample.sample_id]
                    if task.channel == "chain"
                    else None
                ),
            )

        def handle_completed_evaluation_task(task: PendingEvaluationTask) -> None:
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
                        sample_id=task.sample.sample_id,
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

            existing_records_by_sample_id[task.sample.sample_id] = record
            if task.channel == "chain":
                chain_results_by_sample_id[task.sample.sample_id] = record
                completed_chain_sample_ids_this_run.append(task.sample.sample_id)
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
                normal_results_by_sample_id[task.sample.sample_id] = record
                completed_normal_sample_ids_this_run.append(task.sample.sample_id)
                _append_result(results_path, record)

        def drain_completed_structuring_tasks(*, wait_all: bool) -> None:
            sample_ids = (
                list(pending_structuring_tasks)
                if wait_all
                else [
                    sample_id
                    for sample_id, task in pending_structuring_tasks.items()
                    if task.future.done()
                ]
            )
            for sample_id in sample_ids:
                task = pending_structuring_tasks.pop(sample_id)
                handle_completed_structuring_task(task)

        def drain_completed_evaluation_tasks(*, wait_all: bool) -> None:
            sample_ids = (
                list(pending_evaluation_tasks)
                if wait_all
                else [
                    sample_id
                    for sample_id, task in pending_evaluation_tasks.items()
                    if task.future.done()
                ]
            )
            for sample_id in sample_ids:
                task = pending_evaluation_tasks.pop(sample_id)
                handle_completed_evaluation_task(task)

        def drain_pending(*, wait_all: bool) -> None:
            drain_completed_structuring_tasks(wait_all=wait_all)
            drain_completed_evaluation_tasks(wait_all=wait_all)

        for sample in normal_target_samples:
            if sample.sample_id in existing_records_by_sample_id:
                drain_pending(wait_all=False)
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
                    drain_pending(wait_all=False)
                    continue
                normal_prediction_map[sample.sample_id] = raw_output
                _append_prediction_artifact(
                    predictions_path,
                    sample.sample_id,
                    raw_output,
                    config.protocol,
                )

            if sample.sample_id in pending_evaluation_tasks:
                drain_pending(wait_all=False)
                continue

            structured_record = normal_structured_by_sample_id.get(sample.sample_id)
            if structured_record is not None:
                pending_evaluation_tasks[sample.sample_id] = submit_evaluation(
                    sample=sample,
                    structured_record=structured_record,
                    channel="normal",
                )
            elif sample.sample_id not in pending_structuring_tasks:
                pending_structuring_tasks[sample.sample_id] = submit_structuring(
                    sample=sample,
                    raw_output=raw_output,
                    channel="normal",
                )
            drain_pending(wait_all=False)

        drain_pending(wait_all=False)

        for pair in chain_pairs:
            downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
            if downstream_sample.sample_id in existing_records_by_sample_id:
                drain_pending(wait_all=False)
                continue

            upstream_raw_output = normal_prediction_map.get(pair.upstream_sample_id)
            if upstream_raw_output is None:
                blocked_chain_sample_ids.append(pair.downstream_sample_id)
                drain_pending(wait_all=False)
                continue

            raw_output = chain_prediction_map.get(pair.downstream_sample_id)
            if raw_output is None:
                try:
                    raw_output = adapter.predict(
                        build_model_input(
                            downstream_sample,
                            rendered_prompts_by_sample_id[downstream_sample.sample_id],
                            conversation_history=build_chain_history(
                                rendered_prompts_by_sample_id[pair.upstream_sample_id].prompt_text,
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
                    drain_pending(wait_all=False)
                    continue
                chain_prediction_map[pair.downstream_sample_id] = raw_output
                _append_chain_prediction_artifact(
                    chain_predictions_path,
                    pair,
                    raw_output,
                    config.protocol,
                )

            if pair.downstream_sample_id in pending_evaluation_tasks:
                drain_pending(wait_all=False)
                continue

            structured_record = chain_structured_by_sample_id.get(pair.downstream_sample_id)
            if structured_record is not None:
                pending_evaluation_tasks[pair.downstream_sample_id] = submit_evaluation(
                    sample=downstream_sample,
                    structured_record=structured_record,
                    channel="chain",
                    pair=pair,
                )
            elif pair.downstream_sample_id not in pending_structuring_tasks:
                pending_structuring_tasks[pair.downstream_sample_id] = submit_structuring(
                    sample=downstream_sample,
                    raw_output=raw_output,
                    channel="chain",
                    pair=pair,
                )
            drain_pending(wait_all=False)

        drain_completed_structuring_tasks(wait_all=True)
        drain_completed_evaluation_tasks(wait_all=True)

    ordered_records = [
        existing_records_by_sample_id[sample.sample_id]
        for sample in prepared_samples
        if sample.sample_id in existing_records_by_sample_id
    ]
    evaluation = summarize_evaluation_records(
        prepared_samples,
        ordered_records,
        model_name=model_name,
    )

    oracle_summary = None
    oracle_variant_pair_results = None
    oracle_errors_this_run: list[dict[str, str]] = []
    if config.chain_manifest:
        oracle_variant_pair_results = (
            {
                variant: _load_existing_oracle_pair_results(oracle_pair_results_paths[variant])
                for variant in ORACLE_EXPERIMENT_B_VARIANTS
            }
            if config.enable_oracle_track
            else None
        )
        if config.enable_oracle_track:
            for variant in ORACLE_EXPERIMENT_B_VARIANTS:
                variant_pair_results = oracle_variant_pair_results[variant]
                variant_results_path = oracle_pair_results_paths[variant]
                for pair in chain_pairs:
                    if pair.pair_id in variant_pair_results:
                        continue
                    try:
                        pair_result = evaluate_oracle_chain_pair(
                            adapter,
                            prepared_by_sample_id,
                            pair,
                            prompt_pack=prompt_pack,
                            oracle_prompt_pack=oracle_prompt_pack,
                            structurer_service=structurer_service,
                            judge_client=judge_client,
                            variant=variant,
                        )
                    except OraclePairError as exc:
                        oracle_errors_this_run.append(
                            {
                                "variant": variant,
                                "pair_id": exc.pair_id,
                                "stage": exc.stage,
                                "reason": f"{type(exc.cause).__name__}: {exc.cause}",
                            }
                        )
                        continue
                    variant_pair_results[pair.pair_id] = pair_result
                    _append_oracle_pair_result(
                        variant_results_path,
                        pair.pair_id,
                        pair_result["upstream"],
                        pair_result["downstream"],
                    )

        oracle_summary = summarize_experiment_b(
            chain_pairs,
            evaluation["records_by_sample_id"],
            oracle_variant_pair_results=oracle_variant_pair_results,
        )

    write_jsonl(
        predictions_path,
        [
            {"sample_id": sample_id, "raw_output": raw_output, "protocol_id": config.protocol}
            for sample_id, raw_output in sorted(normal_prediction_map.items())
        ],
    )
    write_jsonl(
        structured_predictions_path,
        [
            record.to_dict()
            for sample_id, record in sorted(normal_structured_by_sample_id.items())
            if sample_id in normal_target_sample_ids
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
        chain_structured_predictions_path,
        [
            record.to_dict()
            for sample_id, record in sorted(chain_structured_by_sample_id.items())
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
    if oracle_variant_pair_results is not None:
        for variant in ORACLE_EXPERIMENT_B_VARIANTS:
            write_jsonl(
                oracle_pair_results_paths[variant],
                [
                    {
                        "pair_id": pair_id,
                        "upstream": pair_result["upstream"].to_dict(),
                        "downstream": pair_result["downstream"].to_dict(),
                    }
                    for pair_id, pair_result in sorted(oracle_variant_pair_results[variant].items())
                ],
            )

    protocol_manifest = read_json(config.prepared_root / config.protocol / "build_manifest.json")

    completed_total = len(
        [sample_id for sample_id in existing_records_by_sample_id if sample_id in target_sample_ids]
    )
    normal_predicted_sample_ids = normal_target_sample_ids & set(normal_prediction_map)
    normal_structured_sample_ids = normal_target_sample_ids & set(normal_structured_by_sample_id)
    normal_completed_sample_ids = normal_target_sample_ids & set(normal_results_by_sample_id)
    chain_target_sample_ids = chain_downstream_sample_ids & target_sample_ids
    chain_predicted_sample_ids = chain_target_sample_ids & set(chain_prediction_map)
    chain_structured_sample_ids = chain_target_sample_ids & set(chain_structured_by_sample_id)
    chain_completed_sample_ids = chain_target_sample_ids & set(chain_results_by_sample_id)

    failed_samples_this_run = [
        *prediction_errors_this_run,
        *structuring_errors_this_run,
        *evaluation_errors_this_run,
        *chain_prediction_errors_this_run,
        *chain_structuring_errors_this_run,
        *chain_evaluation_errors_this_run,
    ]

    run_status = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "run_name": run_dir.name,
        "total_target_samples": len(target_samples),
        "completed_samples_before_run": completed_before_run,
        "structured_normal_samples_this_run": len(structured_normal_sample_ids_this_run),
        "completed_normal_samples_this_run": len(completed_normal_sample_ids_this_run),
        "structured_chain_samples_this_run": len(structured_chain_sample_ids_this_run),
        "completed_chain_samples_this_run": len(completed_chain_sample_ids_this_run),
        "completed_samples_total": completed_total,
        "pending_prediction_sample_ids": sorted(normal_target_sample_ids - normal_predicted_sample_ids),
        "predicted_not_structured_sample_ids": sorted(
            normal_predicted_sample_ids - normal_structured_sample_ids
        ),
        "structured_not_evaluated_sample_ids": sorted(
            normal_structured_sample_ids - normal_completed_sample_ids
        ),
        "pending_chain_prediction_sample_ids": sorted(
            chain_target_sample_ids - chain_predicted_sample_ids
        ),
        "chain_predicted_not_structured_sample_ids": sorted(
            chain_predicted_sample_ids - chain_structured_sample_ids
        ),
        "chain_structured_not_evaluated_sample_ids": sorted(
            chain_structured_sample_ids - chain_completed_sample_ids
        ),
        "blocked_chain_sample_ids": sorted(set(blocked_chain_sample_ids)),
        "prediction_errors_this_run": prediction_errors_this_run,
        "structuring_errors_this_run": structuring_errors_this_run,
        "evaluation_errors_this_run": evaluation_errors_this_run,
        "chain_prediction_errors_this_run": chain_prediction_errors_this_run,
        "chain_structuring_errors_this_run": chain_structuring_errors_this_run,
        "chain_evaluation_errors_this_run": chain_evaluation_errors_this_run,
        "failed_samples_this_run": failed_samples_this_run,
        "failed_sample_ids_this_run": sorted(
            {error["sample_id"] for error in failed_samples_this_run}
        ),
        "oracle_errors_this_run": oracle_errors_this_run,
    }
    summary_payload = {
        "model_name": model_name,
        "protocol_id": config.protocol,
        "overall": evaluation["overall"],
        "task_summaries": [summary.to_dict() for summary in evaluation["task_summaries"]],
        "experiment_b": oracle_summary,
        "data_status": protocol_manifest["data_status"],
        "run_status": run_status,
    }
    write_json(summary_path, summary_payload)
    print(
        "Run status: "
        f"{completed_total}/{len(target_samples)} completed, "
        f"{len(run_status['pending_prediction_sample_ids'])} pending prediction, "
        f"{len(run_status['predicted_not_structured_sample_ids'])} predicted but not structured, "
        f"{len(run_status['structured_not_evaluated_sample_ids'])} structured but not evaluated, "
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
