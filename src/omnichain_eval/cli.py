"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import logging
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
from .logging_utils import get_logger, log_event, setup_run_logging
from .metrics import evaluate_sample
from .protocols import resolve_protocol
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

LOGGER = get_logger(__name__)


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


@dataclass(slots=True)
class StageProgressSnapshot:
    predicted: int
    structured: int
    judged: int
    prediction_errors: int
    structuring_errors: int
    evaluation_errors: int
    blocked: int = 0

    @property
    def completed_units(self) -> int:
        return self.predicted + self.structured + self.judged


@dataclass(slots=True)
class StageProgressLogger:
    logger: Any
    event_name: str
    total_samples: int
    last_snapshot: StageProgressSnapshot | None = None

    def log_start(self, **fields: Any) -> None:
        log_event(self.logger, logging.INFO, f"{self.event_name}_start", **fields)

    def log_progress(
        self,
        snapshot: StageProgressSnapshot,
        *,
        force: bool = False,
        event_suffix: str = "progress",
    ) -> None:
        if not force and snapshot == self.last_snapshot:
            return
        self.last_snapshot = snapshot
        log_event(
            self.logger,
            logging.INFO,
            f"{self.event_name}_{event_suffix}",
            prediction=f"{snapshot.predicted}/{self.total_samples}",
            structured=f"{snapshot.structured}/{self.total_samples}",
            judged=f"{snapshot.judged}/{self.total_samples}",
            prediction_errors=snapshot.prediction_errors,
            structuring_errors=snapshot.structuring_errors,
            evaluation_errors=snapshot.evaluation_errors,
            blocked_chain=snapshot.blocked,
        )

    def log_done(self, snapshot: StageProgressSnapshot) -> None:
        self.log_progress(snapshot, force=True, event_suffix="done")


@dataclass(slots=True)
class OracleVariantProgressLogger:
    logger: Any
    variant: str
    total_pairs: int
    last_counts: tuple[int, int] | None = None

    def log_start(self, *, completed_pairs: int, failed_pairs: int) -> None:
        log_event(
            self.logger,
            logging.INFO,
            "oracle_variant_start",
            variant=self.variant,
            completed_pairs=f"{completed_pairs}/{self.total_pairs}",
            failed_pairs=failed_pairs,
            remaining_pairs=max(self.total_pairs - completed_pairs - failed_pairs, 0),
        )

    def log_progress(self, *, completed_pairs: int, failed_pairs: int, force: bool = False) -> None:
        processed_pairs = completed_pairs + failed_pairs
        counts = (completed_pairs, failed_pairs)
        if not force and counts == self.last_counts:
            return
        self.last_counts = counts
        log_event(
            self.logger,
            logging.INFO,
            "oracle_variant_progress",
            variant=self.variant,
            completed_pairs=f"{completed_pairs}/{self.total_pairs}",
            failed_pairs=failed_pairs,
            remaining_pairs=max(self.total_pairs - processed_pairs, 0),
        )

    def log_done(self, *, completed_pairs: int, failed_pairs: int) -> None:
        log_event(
            self.logger,
            logging.INFO,
            "oracle_variant_done",
            variant=self.variant,
            completed_pairs=f"{completed_pairs}/{self.total_pairs}",
            failed_pairs=failed_pairs,
            remaining_pairs=max(self.total_pairs - completed_pairs - failed_pairs, 0),
        )


def _stage_progress_snapshot(
    *,
    target_sample_ids: set[str],
    prediction_map: dict[str, str],
    structured_map: dict[str, StructuredPredictionRecord],
    results_map: dict[str, EvaluationRecord],
    prediction_errors: list[dict[str, str]],
    structuring_errors: list[dict[str, str]],
    evaluation_errors: list[dict[str, str]],
    blocked_sample_ids: list[str] | None = None,
) -> StageProgressSnapshot:
    return StageProgressSnapshot(
        predicted=len(target_sample_ids & set(prediction_map)),
        structured=len(target_sample_ids & set(structured_map)),
        judged=len(target_sample_ids & set(results_map)),
        prediction_errors=len(prediction_errors),
        structuring_errors=len(structuring_errors),
        evaluation_errors=len(evaluation_errors),
        blocked=len(set(blocked_sample_ids or [])),
    )


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
    runtime_protocol = resolve_protocol(config.protocol)
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
        artifacts_root
        / (config.run_name or _default_run_name(model_name, runtime_protocol.protocol_id))
    )
    log_path = run_dir / "run.log"
    setup_run_logging(log_path)
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

    judge_client = _judge_client_from_config(config.judge)

    log_event(
        LOGGER,
        logging.INFO,
        "run_init",
        model_name=model_name,
        protocol_id=config.protocol,
        run_dir=run_dir,
        log_path=log_path,
        total_target_samples=len(target_samples),
        normal_samples=len(normal_target_samples),
        chain_samples=len(chain_pairs),
        oracle_track=config.enable_oracle_track,
        completed_before_run=completed_before_run,
        structurer_concurrency=config.structurer.concurrency,
        judge_concurrency=config.judge.concurrency,
    )
    if completed_before_run:
        log_event(
            LOGGER,
            logging.INFO,
            "run_resume",
            run_dir=run_dir,
            completed_before_run=completed_before_run,
        )

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
    normal_progress_logger = StageProgressLogger(
        logger=LOGGER,
        event_name="normal_stage",
        total_samples=len(normal_target_samples),
    )
    chain_progress_logger = StageProgressLogger(
        logger=LOGGER,
        event_name="chain_stage",
        total_samples=len(chain_pairs),
    )

    with ThreadPoolExecutor(
        max_workers=config.structurer.concurrency,
        thread_name_prefix="omnichain-structurer",
    ) as structurer_executor, ThreadPoolExecutor(
        max_workers=config.judge.concurrency,
        thread_name_prefix="omnichain-judge",
    ) as judge_executor:
        pending_structuring_tasks: dict[str, PendingStructuringTask] = {}
        pending_evaluation_tasks: dict[str, PendingEvaluationTask] = {}

        def log_normal_progress(*, force: bool = False, event_suffix: str = "progress") -> None:
            normal_progress_logger.log_progress(
                _stage_progress_snapshot(
                    target_sample_ids=normal_target_sample_ids,
                    prediction_map=normal_prediction_map,
                    structured_map=normal_structured_by_sample_id,
                    results_map=normal_results_by_sample_id,
                    prediction_errors=prediction_errors_this_run,
                    structuring_errors=structuring_errors_this_run,
                    evaluation_errors=evaluation_errors_this_run,
                ),
                force=force,
                event_suffix=event_suffix,
            )

        def log_chain_progress(*, force: bool = False, event_suffix: str = "progress") -> None:
            chain_progress_logger.log_progress(
                _stage_progress_snapshot(
                    target_sample_ids=chain_downstream_sample_ids,
                    prediction_map=chain_prediction_map,
                    structured_map=chain_structured_by_sample_id,
                    results_map=chain_results_by_sample_id,
                    prediction_errors=chain_prediction_errors_this_run,
                    structuring_errors=chain_structuring_errors_this_run,
                    evaluation_errors=chain_evaluation_errors_this_run,
                    blocked_sample_ids=blocked_chain_sample_ids,
                ),
                force=force,
                event_suffix=event_suffix,
            )

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
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "structuring_error",
                    sample_id=task.sample.sample_id,
                    channel=task.channel,
                    pair_id=task.pair_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if task.channel == "chain":
                    log_chain_progress()
                else:
                    log_normal_progress()
                return

            if task.channel == "chain":
                chain_structured_by_sample_id[task.sample.sample_id] = record
                structured_chain_sample_ids_this_run.append(task.sample.sample_id)
                _append_structured_artifact(chain_structured_predictions_path, record)
                log_chain_progress()
            else:
                normal_structured_by_sample_id[task.sample.sample_id] = record
                structured_normal_sample_ids_this_run.append(task.sample.sample_id)
                _append_structured_artifact(structured_predictions_path, record)
                log_normal_progress()

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
                log_event(
                    LOGGER,
                    logging.ERROR,
                    "evaluation_error",
                    sample_id=task.sample.sample_id,
                    channel=task.channel,
                    pair_id=task.pair_id,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if task.channel == "chain":
                    log_chain_progress()
                else:
                    log_normal_progress()
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
                log_chain_progress()
            else:
                normal_results_by_sample_id[task.sample.sample_id] = record
                completed_normal_sample_ids_this_run.append(task.sample.sample_id)
                _append_result(results_path, record)
                log_normal_progress()

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

        normal_progress_logger.log_start(
            total_samples=len(normal_target_samples),
            resumed_prediction=len(normal_target_sample_ids & set(normal_prediction_map)),
            resumed_structured=len(normal_target_sample_ids & set(normal_structured_by_sample_id)),
            resumed_judged=len(normal_target_sample_ids & set(normal_results_by_sample_id)),
        )
        log_normal_progress(force=True)
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
                    log_event(
                        LOGGER,
                        logging.ERROR,
                        "prediction_error",
                        sample_id=sample.sample_id,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    log_normal_progress()
                    drain_pending(wait_all=False)
                    continue
                normal_prediction_map[sample.sample_id] = raw_output
                _append_prediction_artifact(
                    predictions_path,
                    sample.sample_id,
                    raw_output,
                    config.protocol,
                )
                log_normal_progress()

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

        chain_progress_logger.log_start(
            total_samples=len(chain_pairs),
            resumed_prediction=len(chain_downstream_sample_ids & set(chain_prediction_map)),
            resumed_structured=len(chain_downstream_sample_ids & set(chain_structured_by_sample_id)),
            resumed_judged=len(chain_downstream_sample_ids & set(chain_results_by_sample_id)),
        )
        log_chain_progress(force=True)
        for pair in chain_pairs:
            downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
            if downstream_sample.sample_id in existing_records_by_sample_id:
                drain_pending(wait_all=False)
                continue

            upstream_raw_output = normal_prediction_map.get(pair.upstream_sample_id)
            if upstream_raw_output is None:
                if pair.downstream_sample_id not in blocked_chain_sample_ids:
                    blocked_chain_sample_ids.append(pair.downstream_sample_id)
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "chain_blocked",
                        sample_id=pair.downstream_sample_id,
                        upstream_sample_id=pair.upstream_sample_id,
                        pair_id=pair.pair_id,
                        reason="missing_upstream_prediction",
                    )
                    log_chain_progress()
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
                    log_event(
                        LOGGER,
                        logging.ERROR,
                        "chain_prediction_error",
                        sample_id=pair.downstream_sample_id,
                        pair_id=pair.pair_id,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    log_chain_progress()
                    drain_pending(wait_all=False)
                    continue
                chain_prediction_map[pair.downstream_sample_id] = raw_output
                _append_chain_prediction_artifact(
                    chain_predictions_path,
                    pair,
                    raw_output,
                    config.protocol,
                )
                log_chain_progress()

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
        log_normal_progress(force=True, event_suffix="done")
        log_chain_progress(force=True, event_suffix="done")

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
            log_event(
                LOGGER,
                logging.INFO,
                "oracle_stage_start",
                total_pairs=len(chain_pairs),
                variants=list(ORACLE_EXPERIMENT_B_VARIANTS),
            )
            for variant in ORACLE_EXPERIMENT_B_VARIANTS:
                variant_pair_results = oracle_variant_pair_results[variant]
                variant_results_path = oracle_pair_results_paths[variant]
                variant_failed_pairs = 0
                variant_progress_logger = OracleVariantProgressLogger(
                    logger=LOGGER,
                    variant=variant,
                    total_pairs=len(chain_pairs),
                )
                variant_progress_logger.log_start(
                    completed_pairs=len(variant_pair_results),
                    failed_pairs=variant_failed_pairs,
                )
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
                        variant_failed_pairs += 1
                        log_event(
                            LOGGER,
                            logging.ERROR,
                            "oracle_pair_error",
                            variant=variant,
                            pair_id=exc.pair_id,
                            stage=exc.stage,
                            reason=f"{type(exc.cause).__name__}: {exc.cause}",
                        )
                        variant_progress_logger.log_progress(
                            completed_pairs=len(variant_pair_results),
                            failed_pairs=variant_failed_pairs,
                        )
                        continue
                    variant_pair_results[pair.pair_id] = pair_result
                    _append_oracle_pair_result(
                        variant_results_path,
                        pair.pair_id,
                        pair_result["upstream"],
                        pair_result["downstream"],
                    )
                    variant_progress_logger.log_progress(
                        completed_pairs=len(variant_pair_results),
                        failed_pairs=variant_failed_pairs,
                    )
                variant_progress_logger.log_done(
                    completed_pairs=len(variant_pair_results),
                    failed_pairs=variant_failed_pairs,
                )
            log_event(
                LOGGER,
                logging.INFO,
                "oracle_stage_done",
                total_pairs=len(chain_pairs),
                total_errors=len(oracle_errors_this_run),
            )
        else:
            log_event(
                LOGGER,
                logging.INFO,
                "oracle_stage_skipped",
                reason="disabled",
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
    log_event(
        LOGGER,
        logging.INFO,
        "summary_write_start",
        summary_path=summary_path,
        completed_total=completed_total,
        total_target_samples=len(target_samples),
    )
    write_json(summary_path, summary_payload)
    log_event(
        LOGGER,
        logging.INFO,
        "summary_write_done",
        summary_path=summary_path,
        completed_total=completed_total,
        total_target_samples=len(target_samples),
        pending_prediction=len(run_status["pending_prediction_sample_ids"]),
        predicted_not_structured=len(run_status["predicted_not_structured_sample_ids"]),
        structured_not_evaluated=len(run_status["structured_not_evaluated_sample_ids"]),
        pending_chain_prediction=len(run_status["pending_chain_prediction_sample_ids"]),
    )
    log_event(
        LOGGER,
        logging.INFO,
        "run_done",
        run_dir=run_dir,
        summary_path=summary_path,
    )
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
