"""Experiment orchestration helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .adapters.base import BaseModelAdapter
from .constants import (
    BERTSCORE_MODEL,
    ORACLE_EXPERIMENT_B_VARIANTS,
    ORACLE_VARIANT_LANGUAGE,
    ORACLE_VARIANT_LANGUAGE_VISUAL,
    ORACLE_VARIANT_VISUAL,
    TASK_CONTINUOUS_ACTIONS,
)
from .dataset import DatasetScanReport, scan_dataset_report
from .judge import JudgeClient
from .metrics import evaluate_sample, summarize_task_records
from .prompting import (
    PromptTemplate,
    build_chain_history,
    build_model_input,
    render_oracle_upstream_prompt,
    render_prompt,
)
from .prepare import load_prepared_samples
from .schema import (
    ChainPairRecord,
    EvaluationRecord,
    PreparedSample,
    StructuredPredictionRecord,
    TaskSummary,
)
from .structurer import StructurerService
from .utils import read_jsonl, write_jsonl


class OraclePairError(RuntimeError):
    def __init__(self, pair_id: str, stage: str, cause: Exception):
        super().__init__(f"{pair_id} failed at {stage}: {cause}")
        self.pair_id = pair_id
        self.stage = stage
        self.cause = cause


def build_chain_manifest(
    data_root: Path,
    output_path: Path,
    *,
    scan_report: DatasetScanReport | None = None,
) -> list[ChainPairRecord]:
    report = scan_report or scan_dataset_report(data_root)
    if report.issues:
        issue_text = "\n".join(report.issues[:50])
        raise ValueError(
            f"dataset validation failed with {len(report.issues)} issue(s):\n{issue_text}"
        )
    records = report.supported_records
    by_source_file: dict[Path, dict[str, Any]] = defaultdict(dict)
    for record in records:
        by_source_file[record.source_annotation_path][record.annotation_id] = record
    chain_pairs: list[ChainPairRecord] = []
    for record in records:
        if record.task_name != "Spatial_Imagination":
            continue
        upstream = by_source_file[record.source_annotation_path][record.upstream_annotation_id]
        pair = ChainPairRecord(
            pair_id=f"{record.sample_id}|{upstream.sample_id}",
            video_key=record.video_key,
            upstream_sample_id=upstream.sample_id,
            downstream_sample_id=record.sample_id,
            upstream_task_name=upstream.task_name,
        )
        chain_pairs.append(pair)
    chain_pairs.sort(key=lambda item: item.pair_id)
    write_jsonl(output_path, [pair.to_dict() for pair in chain_pairs])
    return chain_pairs


def load_chain_pairs(path: Path) -> list[ChainPairRecord]:
    return [ChainPairRecord(**row) for row in read_jsonl(path)]


def compute_bertscore(records: list[EvaluationRecord]) -> None:
    eligible = [
        record
        for record in records
        if record.bertscore_reference and record.bertscore_candidate
    ]
    if not eligible:
        return
    try:
        from bert_score import score
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "bert-score dependency is not available in the current environment."
        ) from exc
    _, _, f1_scores = score(
        cands=[record.bertscore_candidate for record in eligible],
        refs=[record.bertscore_reference for record in eligible],
        model_type=BERTSCORE_MODEL,
        lang="en",
        rescale_with_baseline=True,
        idf=False,
        use_fast_tokenizer=False,
    )
    for record, score_value in zip(eligible, f1_scores.tolist(), strict=True):
        record.component_metrics["bertscore_f1"] = float(score_value)


def summarize_evaluation_records(
    prepared_samples: list[PreparedSample],
    records: list[EvaluationRecord],
    *,
    model_name: str,
) -> dict[str, Any]:
    tasks_present = defaultdict(list)
    for prepared_sample in prepared_samples:
        tasks_present[prepared_sample.task_name].append(prepared_sample)
    compute_bertscore(records)

    records_by_task: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        records_by_task[record.task_name].append(record)

    task_summaries: list[TaskSummary] = []
    for task_name, task_samples in sorted(tasks_present.items()):
        task_records = records_by_task.get(task_name, [])
        task_summaries.append(
            summarize_task_records(
                model_name,
                task_samples[0].protocol_id,
                task_name,
                task_records,
                total_num_samples=len(task_samples),
            )
        )

    included = [
        summary.task_accuracy
        for summary in task_summaries
        if summary.task_accuracy is not None
    ]
    overall = (sum(included) / len(included)) if included else None
    records_by_sample_id = {record.sample_id: record for record in records}
    ordered_records = [
        records_by_sample_id[prepared_sample.sample_id]
        for prepared_sample in prepared_samples
        if prepared_sample.sample_id in records_by_sample_id
    ]
    return {
        "records": ordered_records,
        "task_summaries": task_summaries,
        "overall": overall,
        "records_by_sample_id": records_by_sample_id,
    }


def evaluate_prepared_predictions(
    prepared_samples: list[PreparedSample],
    structured_prediction_map: dict[str, StructuredPredictionRecord],
    *,
    model_name: str,
    judge_client: JudgeClient | None,
) -> dict[str, Any]:
    records: list[EvaluationRecord] = []
    for prepared_sample in prepared_samples:
        structured_record = structured_prediction_map.get(prepared_sample.sample_id)
        if structured_record is None:
            continue
        records.append(
            evaluate_sample(
                prepared_sample,
                structured_record,
                judge_client=judge_client,
            )
        )
    return summarize_evaluation_records(
        prepared_samples,
        records,
        model_name=model_name,
    )


def evaluate_oracle_chain_pair(
    adapter: BaseModelAdapter,
    prepared_by_sample_id: dict[str, PreparedSample],
    pair: ChainPairRecord,
    *,
    prompt_pack: dict[str, PromptTemplate],
    oracle_prompt_pack: dict[str, dict[str, PromptTemplate]],
    structurer_service: StructurerService,
    judge_client: JudgeClient | None,
    variant: str,
) -> dict[str, EvaluationRecord]:
    upstream_sample = prepared_by_sample_id[pair.upstream_sample_id]
    downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
    oracle_upstream_prompt = render_oracle_upstream_prompt(
        oracle_prompt_pack,
        upstream_sample,
        variant=variant,
    )
    oracle_upstream_sample = _oracle_input_sample_for_variant(upstream_sample, variant=variant)

    try:
        upstream_raw = adapter.predict(
            build_model_input(
                oracle_upstream_sample,
                oracle_upstream_prompt,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_upstream_prediction", exc) from exc

    try:
        upstream_structured = structurer_service.structure(
            upstream_sample,
            upstream_raw,
            oracle_upstream=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_upstream_structuring", exc) from exc

    upstream_record = StructuredPredictionRecord(
        sample_id=upstream_sample.sample_id,
        task_name=upstream_sample.task_name,
        video_key=upstream_sample.video_key,
        protocol_id=upstream_sample.protocol_id,
        raw_output=upstream_structured.raw_output,
        structured_prediction=upstream_structured.structured_prediction,
        structuring_errors=upstream_structured.errors,
        structuring_warnings=upstream_structured.warnings,
        structurer_raw_response=upstream_structured.structurer_raw_response,
        pair_id=pair.pair_id,
        upstream_sample_id=pair.upstream_sample_id,
        downstream_sample_id=pair.downstream_sample_id,
    )

    try:
        oracle_upstream = evaluate_sample(
            upstream_sample,
            upstream_record,
            judge_client=judge_client,
            oracle_upstream=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_upstream_evaluation", exc) from exc

    conversation_history = build_chain_history(
        oracle_upstream_prompt.prompt_text,
        upstream_raw,
    )
    try:
        downstream_raw = adapter.predict(
            build_model_input(
                downstream_sample,
                render_prompt(prompt_pack, downstream_sample),
                conversation_history=conversation_history,
            )
        )
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_downstream_prediction", exc) from exc

    try:
        downstream_structured = structurer_service.structure(downstream_sample, downstream_raw)
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_downstream_structuring", exc) from exc

    downstream_record = StructuredPredictionRecord(
        sample_id=downstream_sample.sample_id,
        task_name=downstream_sample.task_name,
        video_key=downstream_sample.video_key,
        protocol_id=downstream_sample.protocol_id,
        raw_output=downstream_structured.raw_output,
        structured_prediction=downstream_structured.structured_prediction,
        structuring_errors=downstream_structured.errors,
        structuring_warnings=downstream_structured.warnings,
        structurer_raw_response=downstream_structured.structurer_raw_response,
        pair_id=pair.pair_id,
        upstream_sample_id=pair.upstream_sample_id,
        downstream_sample_id=pair.downstream_sample_id,
    )

    try:
        oracle_downstream = evaluate_sample(
            downstream_sample,
            downstream_record,
            judge_client=judge_client,
        )
    except Exception as exc:  # noqa: BLE001
        raise OraclePairError(pair.pair_id, "oracle_downstream_evaluation", exc) from exc

    return {
        "upstream": oracle_upstream,
        "downstream": oracle_downstream,
    }


def _oracle_input_sample_for_variant(sample: PreparedSample, *, variant: str) -> PreparedSample:
    if variant == ORACLE_VARIANT_LANGUAGE:
        return sample
    if variant in {ORACLE_VARIANT_VISUAL, ORACLE_VARIANT_LANGUAGE_VISUAL}:
        if not sample.oracle_visual_frame_files:
            raise ValueError(f"{sample.sample_id}: missing oracle_visual_frame_files")
        if sample.oracle_visual_sampled_video_file is None:
            raise ValueError(f"{sample.sample_id}: missing oracle_visual_sampled_video_file")
        return replace(
            sample,
            frame_files=list(sample.oracle_visual_frame_files),
            sampled_video_file=sample.oracle_visual_sampled_video_file,
        )
    raise ValueError(f"unsupported Oracle variant: {variant}")


def _summarize_base_chain_pairs(
    chain_pairs: list[ChainPairRecord],
    records_by_sample_id: dict[str, EvaluationRecord],
) -> dict[str, Any]:
    understanding_values: list[int] = []
    reasoning_values: list[int] = []
    chain_values: list[int] = []
    chain_wo_track_values: list[int] = []
    pending_pairs = 0

    for pair in chain_pairs:
        upstream = records_by_sample_id.get(pair.upstream_sample_id)
        downstream = records_by_sample_id.get(pair.downstream_sample_id)
        if upstream is None or downstream is None:
            pending_pairs += 1
            continue
        tracking_pass = int(upstream.component_pass.get("tracking_pass", 0))
        if pair.upstream_task_name == TASK_CONTINUOUS_ACTIONS:
            understanding_non_tracking = int(upstream.component_pass.get("judge_pass", 0))
        else:
            understanding_non_tracking = int(upstream.component_pass.get("tiou_pass", 0))
        reasoning_pass = int(downstream.component_pass.get("judge_pass", 0))
        understanding_pass = tracking_pass * understanding_non_tracking
        understanding_values.append(understanding_pass)
        reasoning_values.append(reasoning_pass)
        chain_values.append(understanding_pass * reasoning_pass)
        chain_wo_track_values.append(understanding_non_tracking * reasoning_pass)

    num_scored_pairs = len(understanding_values)
    return {
        "num_chain_samples": len(chain_pairs),
        "num_scored_chain_samples": num_scored_pairs,
        "num_pending_chain_samples": pending_pairs,
        "understanding_acc": (
            sum(understanding_values) / num_scored_pairs if num_scored_pairs else None
        ),
        "reasoning_acc": sum(reasoning_values) / num_scored_pairs if num_scored_pairs else None,
        "chain_success": sum(chain_values) / num_scored_pairs if num_scored_pairs else None,
        "chain_success_wo_track": (
            sum(chain_wo_track_values) / num_scored_pairs if num_scored_pairs else None
        ),
    }


def _summarize_oracle_variant_chain_pairs(
    chain_pairs: list[ChainPairRecord],
    pair_results: dict[str, dict[str, EvaluationRecord]],
) -> dict[str, Any]:
    understanding_values: list[int] = []
    reasoning_values: list[int] = []
    chain_wo_track_values: list[int] = []
    pending_pairs = 0

    for pair in chain_pairs:
        oracle_pair = pair_results.get(pair.pair_id)
        if oracle_pair is None:
            pending_pairs += 1
            continue
        oracle_upstream = oracle_pair["upstream"]
        oracle_downstream = oracle_pair["downstream"]
        if pair.upstream_task_name == TASK_CONTINUOUS_ACTIONS:
            oracle_understanding_non_tracking = int(
                oracle_upstream.component_pass.get("judge_pass", 0)
            )
        else:
            oracle_understanding_non_tracking = int(
                oracle_upstream.component_pass.get("tiou_pass", 0)
            )
        oracle_reasoning_pass = int(oracle_downstream.component_pass.get("judge_pass", 0))
        understanding_values.append(oracle_understanding_non_tracking)
        reasoning_values.append(oracle_reasoning_pass)
        chain_wo_track_values.append(
            oracle_understanding_non_tracking * oracle_reasoning_pass
        )

    num_scored_pairs = len(understanding_values)
    return {
        "num_chain_samples": len(chain_pairs),
        "num_scored_chain_samples": num_scored_pairs,
        "num_pending_chain_samples": pending_pairs,
        "understanding_acc": (
            sum(understanding_values) / num_scored_pairs if num_scored_pairs else None
        ),
        "reasoning_acc": sum(reasoning_values) / num_scored_pairs if num_scored_pairs else None,
        "chain_success_wo_track": (
            sum(chain_wo_track_values) / num_scored_pairs if num_scored_pairs else None
        ),
    }


def summarize_experiment_b(
    chain_pairs: list[ChainPairRecord],
    records_by_sample_id: dict[str, EvaluationRecord],
    *,
    oracle_variant_pair_results: dict[str, dict[str, dict[str, EvaluationRecord]]] | None = None,
) -> dict[str, Any]:
    if not chain_pairs:
        return {}
    summary = {
        "base": _summarize_base_chain_pairs(chain_pairs, records_by_sample_id),
    }
    if oracle_variant_pair_results is not None:
        for variant in ORACLE_EXPERIMENT_B_VARIANTS:
            summary[f"oracle_{variant}"] = _summarize_oracle_variant_chain_pairs(
                chain_pairs,
                oracle_variant_pair_results.get(variant, {}),
            )
    return summary


def load_prepared_for_protocol(prepared_root: Path, protocol_id: str) -> list[PreparedSample]:
    return load_prepared_samples(prepared_root, protocol_id)
