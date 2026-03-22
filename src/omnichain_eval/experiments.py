"""Experiment orchestration and reporting helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .adapters.base import BaseModelAdapter
from .constants import (
    BERTSCORE_MODEL,
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_STG,
    TASKS_EXCLUDED_FROM_OVERALL,
)
from .dataset import scan_dataset
from .judge import JudgeClient
from .metrics import evaluate_sample, summarize_task_records
from .prepare import load_prepared_samples
from .schema import ChainPairRecord, EvaluationRecord, PreparedSample, TaskSummary
from .utils import read_jsonl, write_jsonl


def build_chain_manifest(data_root: Path, output_path: Path) -> list[ChainPairRecord]:
    records, _issues = scan_dataset(data_root)
    by_sample_id = {record.sample_id: record for record in records}
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
            "bert-score is not installed. Install the 'bertscore' extra to enable it."
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


def evaluate_prepared_predictions(
    prepared_samples: list[PreparedSample],
    prediction_map: dict[str, Any],
    *,
    model_name: str,
    judge_client: JudgeClient | None,
    commentary_supported: bool = True,
    enable_bertscore: bool = False,
) -> dict[str, Any]:
    records: list[EvaluationRecord] = []
    tasks_present = defaultdict(list)
    for prepared_sample in prepared_samples:
        tasks_present[prepared_sample.task_name].append(prepared_sample)
        if prepared_sample.task_name == TASK_COMMENTARY and not commentary_supported:
            continue
        raw_output = prediction_map.get(prepared_sample.sample_id)
        records.append(
            evaluate_sample(
                prepared_sample,
                raw_output,
                judge_client=judge_client,
            )
        )
    if enable_bertscore:
        compute_bertscore(records)

    records_by_task: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        records_by_task[record.task_name].append(record)

    task_summaries: list[TaskSummary] = []
    for task_name, task_samples in sorted(tasks_present.items()):
        task_records = records_by_task.get(task_name, [])
        if task_name == TASK_COMMENTARY and not commentary_supported:
            task_summaries.append(
                TaskSummary(
                    model_name=model_name,
                    task_name=task_name,
                    protocol_id=task_samples[0].protocol_id,
                    num_samples=len(task_samples),
                    task_accuracy=None,
                )
            )
            continue
        task_summaries.append(
            summarize_task_records(
                model_name,
                task_samples[0].protocol_id,
                task_name,
                task_records,
            )
        )

    included = [
        summary.task_accuracy
        for summary in task_summaries
        if summary.task_name not in TASKS_EXCLUDED_FROM_OVERALL and summary.task_accuracy is not None
    ]
    overall = (sum(included) / len(included)) if included else None
    return {
        "records": records,
        "task_summaries": task_summaries,
        "overall": overall,
        "records_by_sample_id": {record.sample_id: record for record in records},
    }


def evaluate_oracle_chain(
    adapter: BaseModelAdapter,
    prepared_by_sample_id: dict[str, PreparedSample],
    chain_pairs: list[ChainPairRecord],
    *,
    judge_client: JudgeClient | None,
) -> dict[str, dict[str, EvaluationRecord]]:
    pair_results: dict[str, dict[str, EvaluationRecord]] = {}
    for pair in chain_pairs:
        upstream_sample = prepared_by_sample_id[pair.upstream_sample_id]
        downstream_sample = prepared_by_sample_id[pair.downstream_sample_id]
        upstream_raw = adapter.predict(
            upstream_sample,
            oracle_track=True,
            context={"chain_pair": pair.to_dict(), "role": "upstream"},
        )
        downstream_raw = adapter.predict(
            downstream_sample,
            oracle_track=True,
            context={"chain_pair": pair.to_dict(), "role": "downstream"},
        )
        pair_results[pair.pair_id] = {
            "upstream": evaluate_sample(
                upstream_sample,
                upstream_raw,
                judge_client=judge_client,
                override_tracking_with_gt=upstream_sample.task_name
                in {TASK_CONTINUOUS_ACTIONS, TASK_STG},
            ),
            "downstream": evaluate_sample(
                downstream_sample,
                downstream_raw,
                judge_client=judge_client,
            ),
        }
    return pair_results


def summarize_experiment_b(
    chain_pairs: list[ChainPairRecord],
    records_by_sample_id: dict[str, EvaluationRecord],
    *,
    oracle_pair_results: dict[str, dict[str, EvaluationRecord]] | None = None,
) -> dict[str, Any]:
    if not chain_pairs:
        return {}
    understanding_values: list[int] = []
    reasoning_values: list[int] = []
    chain_values: list[int] = []
    oracle_understanding_values: list[int] = []
    oracle_reasoning_values: list[int] = []
    oracle_chain_values: list[int] = []

    for pair in chain_pairs:
        upstream = records_by_sample_id[pair.upstream_sample_id]
        downstream = records_by_sample_id[pair.downstream_sample_id]
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

        if oracle_pair_results is None:
            continue
        oracle_upstream = oracle_pair_results[pair.pair_id]["upstream"]
        oracle_downstream = oracle_pair_results[pair.pair_id]["downstream"]
        oracle_tracking_pass = int(oracle_upstream.component_pass.get("tracking_pass", 0))
        if pair.upstream_task_name == TASK_CONTINUOUS_ACTIONS:
            oracle_understanding_non_tracking = int(
                oracle_upstream.component_pass.get("judge_pass", 0)
            )
        else:
            oracle_understanding_non_tracking = int(
                oracle_upstream.component_pass.get("tiou_pass", 0)
            )
        oracle_reasoning_pass = int(oracle_downstream.component_pass.get("judge_pass", 0))
        oracle_understanding = oracle_tracking_pass * oracle_understanding_non_tracking
        oracle_understanding_values.append(oracle_understanding)
        oracle_reasoning_values.append(oracle_reasoning_pass)
        oracle_chain_values.append(oracle_understanding * oracle_reasoning_pass)

    summary = {
        "num_chain_samples": len(chain_pairs),
        "understanding_acc": sum(understanding_values) / len(chain_pairs),
        "reasoning_acc": sum(reasoning_values) / len(chain_pairs),
        "chain_success": sum(chain_values) / len(chain_pairs),
    }
    if oracle_pair_results is not None:
        summary.update(
            {
                "understanding_acc_oracle": sum(oracle_understanding_values) / len(chain_pairs),
                "reasoning_acc_oracle": sum(oracle_reasoning_values) / len(chain_pairs),
                "chain_success_oracle": sum(oracle_chain_values) / len(chain_pairs),
            }
        )
    return summary


def load_prepared_for_protocol(prepared_root: Path, protocol_id: str) -> list[PreparedSample]:
    return load_prepared_samples(prepared_root, protocol_id)
