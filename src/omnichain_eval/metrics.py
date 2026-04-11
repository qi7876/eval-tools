"""Metric computation and task pass logic."""

from __future__ import annotations

import statistics
from typing import Any

from .constants import (
    BBOX_IOU_THRESHOLD,
    JUDGE_REQUIRED_TASKS,
    TASK_AI_COACH,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_MULTIPLE,
    TASK_SCOREBOARD_SINGLE,
    TASK_SCORE_PREDICTION,
    TASK_SPATIAL_IMAGINATION,
    TASK_STG,
    TASK_TEMPORAL_CAUSAL,
    TEXT_ONLY_TASKS,
    TIOU_THRESHOLD,
    TRACKING_THRESHOLD,
)
from .judge import JudgeClient, default_judge_fail
from .schema import (
    EvaluationRecord,
    JudgeDecision,
    PreparedSample,
    StructuredPredictionRecord,
    TaskSummary,
)


def bbox_iou(box_pred: list[float], box_gt: list[float]) -> float:
    pred_x1, pred_y1, pred_x2, pred_y2 = box_pred
    gt_x1, gt_y1, gt_x2, gt_y2 = box_gt
    inter_w = max(0.0, min(pred_x2, gt_x2) - max(pred_x1, gt_x1))
    inter_h = max(0.0, min(pred_y2, gt_y2) - max(pred_y1, gt_y1))
    inter_area = inter_w * inter_h
    pred_area = max(0.0, pred_x2 - pred_x1) * max(0.0, pred_y2 - pred_y1)
    gt_area = max(0.0, gt_x2 - gt_x1) * max(0.0, gt_y2 - gt_y1)
    denominator = pred_area + gt_area - inter_area
    if denominator <= 0:
        return 0.0
    return inter_area / denominator


def mot_bbox_to_corner(box: list[float]) -> list[float]:
    left, top, width, height = box
    return [left, top, left + width, top + height]


def temporal_iou(pred_interval: list[int], gt_interval: list[int]) -> float:
    pred_start, pred_end = pred_interval
    gt_start, gt_end = gt_interval
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start) + 1)
    pred_length = pred_end - pred_start + 1
    gt_length = gt_end - gt_start + 1
    union = pred_length + gt_length - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _judge_required(task_name: str) -> bool:
    return task_name in JUDGE_REQUIRED_TASKS


def _closed_factual_pass(decision: JudgeDecision) -> int:
    return int(
        decision.correctness == 1
        and decision.completeness == 1
        and decision.faithfulness == 1
        and decision.final_pass == 1
    )


def _judge_reference_payload(task_name: str, reference_payload: dict[str, Any]) -> dict[str, Any]:
    if task_name in TEXT_ONLY_TASKS | {TASK_AI_COACH, TASK_SCOREBOARD_SINGLE, TASK_OBJECTS_SPATIAL}:
        return {"text": reference_payload["text"]}
    if task_name in {TASK_CONTINUOUS_ACTIONS, TASK_CONTINUOUS_EVENTS}:
        return {"reference_segments": reference_payload["segments_sampled"]}
    raise ValueError(f"unsupported judge reference payload for {task_name}")


def _judge_prediction_payload(
    task_name: str,
    structured_prediction: dict[str, Any],
) -> dict[str, Any]:
    if task_name in TEXT_ONLY_TASKS | {TASK_AI_COACH, TASK_SCOREBOARD_SINGLE, TASK_OBJECTS_SPATIAL}:
        return {"text": structured_prediction["text"]}
    if task_name in {TASK_CONTINUOUS_ACTIONS, TASK_CONTINUOUS_EVENTS}:
        return {"prediction_segments": structured_prediction["segments"]}
    raise ValueError(f"unsupported judge prediction payload for {task_name}")


def _sampled_window_for_metric(
    structured_prediction: dict[str, Any],
) -> list[int] | None:
    window = structured_prediction.get("time_window_sampled")
    if not isinstance(window, list) or len(window) != 2:
        return None
    start = int(window[0])
    end = int(window[1])
    return [start, end]


def _collapse_tracking(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, list[float]], list[str]]:
    collapsed: dict[int, list[float]] = {}
    warnings: list[str] = []
    for row in rows:
        frame_sampled = int(row["frame_sampled"])
        if frame_sampled in collapsed:
            warnings.append(f"duplicate tracking prediction for sampled frame {frame_sampled}")
            continue
        collapsed[frame_sampled] = [float(value) for value in row["bbox_mot"]]
    return collapsed, warnings


def _evaluate_tracking(
    structured_prediction: dict[str, Any],
    gt_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    predicted_rows = structured_prediction.get("tracking", [])
    predicted_by_frame, warnings = _collapse_tracking(predicted_rows)
    if not gt_rows:
        return (
            {"tracking_mean_iou": 0.0, "tracking_pass_rate": 0.0},
            {"tracking_pass": 0},
            warnings,
        )
    frame_ious: list[float] = []
    for gt_row in gt_rows:
        frame_sampled = int(gt_row["frame_sampled"])
        predicted = predicted_by_frame.get(frame_sampled)
        if predicted is None:
            frame_ious.append(0.0)
            continue
        frame_ious.append(
            bbox_iou(mot_bbox_to_corner(predicted), mot_bbox_to_corner(gt_row["bbox_mot"]))
        )
    mean_iou = sum(frame_ious) / len(frame_ious)
    pass_rate = sum(1 for value in frame_ious if value >= TRACKING_THRESHOLD) / len(frame_ious)
    passes = int(mean_iou >= TRACKING_THRESHOLD and pass_rate >= TRACKING_THRESHOLD)
    return (
        {"tracking_mean_iou": mean_iou, "tracking_pass_rate": pass_rate},
        {"tracking_pass": passes},
        warnings,
    )


def _evaluate_object_spatial(
    structured_prediction: dict[str, Any],
    gt_objects: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    predicted_objects = structured_prediction.get("objects", [])
    predicted_by_label = {
        str(row["label"]).strip(): [float(value) for value in row["bbox"]]
        for row in predicted_objects
        if (
            isinstance(row, dict)
            and "label" in row
            and isinstance(row.get("bbox"), list)
            and len(row["bbox"]) == 4
        )
    }
    object_ious: dict[str, float] = {}
    object_passes: dict[str, int] = {}
    for gt_object in gt_objects:
        label = str(gt_object["label"]).strip()
        predicted_bbox = predicted_by_label.get(label)
        if predicted_bbox is None:
            iou = 0.0
        else:
            iou = bbox_iou(predicted_bbox, gt_object["bbox"])
        object_ious[label] = iou
        object_passes[label] = int(iou >= BBOX_IOU_THRESHOLD)
    return {"object_ious": object_ious}, object_passes


def _linearize_segments(segments: list[dict[str, Any]]) -> str:
    return " ".join(segment["text"] for segment in segments)


def _bertscore_pair(
    task_name: str,
    reference_payload: dict[str, Any],
    structured_prediction: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if structured_prediction is None:
        return None, None
    if task_name in TEXT_ONLY_TASKS | {TASK_AI_COACH, TASK_SCOREBOARD_SINGLE, TASK_OBJECTS_SPATIAL}:
        return reference_payload["text"], structured_prediction["text"]
    if task_name in {TASK_CONTINUOUS_ACTIONS, TASK_CONTINUOUS_EVENTS}:
        return (
            _linearize_segments(reference_payload["segments_sampled"]),
            _linearize_segments(structured_prediction["segments"]),
        )
    return None, None


def _judge_textual_task(
    task_name: str,
    prepared_sample: PreparedSample,
    structured_prediction: dict[str, Any] | None,
    judge_client: JudgeClient | None,
) -> tuple[JudgeDecision, int]:
    if structured_prediction is None or judge_client is None:
        return default_judge_fail("missing structured prediction or judge backend"), 0
    if task_name in TEXT_ONLY_TASKS | {TASK_AI_COACH, TASK_SCOREBOARD_SINGLE, TASK_OBJECTS_SPATIAL}:
        if not str(structured_prediction.get("text", "")).strip():
            return default_judge_fail("missing structured text prediction"), 0
    reference_payload = _judge_reference_payload(task_name, prepared_sample.reference_payload)
    prediction_payload = _judge_prediction_payload(
        task_name,
        structured_prediction,
    )
    decision = judge_client.judge(
        task_name,
        prepared_sample.question_text,
        reference_payload,
        prediction_payload,
    )
    if task_name in {
        TASK_SCOREBOARD_SINGLE,
        TASK_SCOREBOARD_MULTIPLE,
        TASK_OBJECTS_SPATIAL,
        TASK_SPATIAL_IMAGINATION,
        TASK_SCORE_PREDICTION,
    }:
        return decision, _closed_factual_pass(decision)
    if task_name == TASK_TEMPORAL_CAUSAL:
        return decision, int(decision.final_pass == 1)
    return decision, int(decision.final_pass == 1)


def evaluate_sample(
    prepared_sample: PreparedSample,
    structured_record: StructuredPredictionRecord,
    *,
    judge_client: JudgeClient | None,
    oracle_upstream: bool = False,
) -> EvaluationRecord:
    structured = structured_record.structured_prediction
    metrics: dict[str, Any] = {}
    component_pass: dict[str, Any] = {}
    judge_decision: JudgeDecision | None = None
    warnings = list(structured_record.structuring_warnings)
    structuring_errors = list(structured_record.structuring_errors)

    if prepared_sample.task_name == TASK_SCOREBOARD_SINGLE:
        predicted_bbox = structured.get("bbox") if structured else None
        if structured and isinstance(predicted_bbox, list) and len(predicted_bbox) == 4:
            iou = bbox_iou(predicted_bbox, prepared_sample.reference_payload["bbox"])
        else:
            iou = 0.0
        metrics["bbox_iou"] = iou
        component_pass["bbox_pass"] = int(iou >= BBOX_IOU_THRESHOLD)
        judge_decision, judge_pass = _judge_textual_task(
            prepared_sample.task_name,
            prepared_sample,
            structured,
            judge_client,
        )
        component_pass["judge_pass"] = judge_pass
        task_pass = int(component_pass["bbox_pass"] and judge_pass)
    elif prepared_sample.task_name == TASK_OBJECTS_SPATIAL:
        object_metrics, object_passes = _evaluate_object_spatial(
            structured or {},
            prepared_sample.reference_payload["objects"],
        )
        metrics.update(object_metrics)
        component_pass["object_passes"] = object_passes
        judge_decision, judge_pass = _judge_textual_task(
            prepared_sample.task_name,
            prepared_sample,
            structured,
            judge_client,
        )
        component_pass["judge_pass"] = judge_pass
        task_pass = int(judge_pass and all(object_passes.values()))
    elif prepared_sample.task_name in TEXT_ONLY_TASKS:
        judge_decision, judge_pass = _judge_textual_task(
            prepared_sample.task_name,
            prepared_sample,
            structured,
            judge_client,
        )
        component_pass["judge_pass"] = judge_pass
        task_pass = judge_pass
    elif prepared_sample.task_name == TASK_CONTINUOUS_EVENTS:
        judge_decision, judge_pass = _judge_textual_task(
            prepared_sample.task_name,
            prepared_sample,
            structured,
            judge_client,
        )
        component_pass["judge_pass"] = judge_pass
        task_pass = judge_pass
    elif prepared_sample.task_name == TASK_CONTINUOUS_ACTIONS:
        judge_decision, judge_pass = _judge_textual_task(
            prepared_sample.task_name,
            prepared_sample,
            structured,
            judge_client,
        )
        component_pass["judge_pass"] = judge_pass
        if oracle_upstream:
            task_pass = judge_pass
        else:
            tracking_metrics, tracking_pass, tracking_warnings = _evaluate_tracking(
                structured or {},
                prepared_sample.reference_payload.get("tracking_gt_sampled", []),
            )
            metrics.update(tracking_metrics)
            component_pass.update(tracking_pass)
            warnings.extend(tracking_warnings)
            task_pass = int(judge_pass and component_pass["tracking_pass"])
    elif prepared_sample.task_name == TASK_STG:
        predicted_sampled_window = None
        if structured is not None:
            predicted_sampled_window = _sampled_window_for_metric(structured)
        if predicted_sampled_window is None:
            tiou = 0.0
        else:
            tiou = temporal_iou(
                predicted_sampled_window,
                prepared_sample.reference_payload["time_window_sampled"],
            )
        metrics["tiou"] = tiou
        component_pass["tiou_pass"] = int(tiou >= TIOU_THRESHOLD)
        if oracle_upstream:
            task_pass = int(component_pass["tiou_pass"])
        else:
            tracking_metrics, tracking_pass, tracking_warnings = _evaluate_tracking(
                structured or {},
                prepared_sample.reference_payload.get("tracking_gt_sampled", []),
            )
            metrics.update(tracking_metrics)
            component_pass.update(tracking_pass)
            warnings.extend(tracking_warnings)
            task_pass = int(component_pass["tiou_pass"] and component_pass["tracking_pass"])
    else:
        raise ValueError(f"unsupported task for evaluation: {prepared_sample.task_name}")

    reference_text, prediction_text = _bertscore_pair(
        prepared_sample.task_name,
        prepared_sample.reference_payload,
        structured,
    )

    return EvaluationRecord(
        sample_id=prepared_sample.sample_id,
        task_name=prepared_sample.task_name,
        video_key=prepared_sample.video_key,
        protocol_id=prepared_sample.protocol_id,
        structured_prediction=structured,
        structuring_errors=structuring_errors,
        structuring_warnings=warnings,
        component_metrics=metrics,
        component_pass=component_pass,
        task_pass=task_pass,
        judge_decision=judge_decision.to_dict() if isinstance(judge_decision, JudgeDecision) else judge_decision,
        raw_output=structured_record.raw_output,
        bertscore_candidate=prediction_text,
        bertscore_reference=reference_text,
    )


def _mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def summarize_task_records(
    model_name: str,
    protocol_id: str,
    task_name: str,
    records: list[EvaluationRecord],
    *,
    total_num_samples: int,
) -> TaskSummary:
    if not records:
        return TaskSummary(
            model_name=model_name,
            task_name=task_name,
            protocol_id=protocol_id,
            num_samples=total_num_samples,
            task_accuracy=None,
            num_scored_samples=0,
            num_pending_samples=total_num_samples,
        )
    judge_passes = [
        float(record.component_pass["judge_pass"])
        for record in records
        if "judge_pass" in record.component_pass
    ]
    bbox_passes = [
        float(value)
        for record in records
        for key, value in record.component_pass.items()
        if key == "bbox_pass"
    ]
    bbox_passes.extend(
        float(value)
        for record in records
        for key, value in record.component_pass.items()
        if key == "object_passes" and isinstance(value, dict)
        for value in value.values()
    )
    tiou_passes = [
        float(record.component_pass["tiou_pass"])
        for record in records
        if "tiou_pass" in record.component_pass
    ]
    tracking_mean_ious = [
        float(record.component_metrics["tracking_mean_iou"])
        for record in records
        if "tracking_mean_iou" in record.component_metrics
    ]
    tracking_pass_rates = [
        float(record.component_metrics["tracking_pass_rate"])
        for record in records
        if "tracking_pass_rate" in record.component_metrics
    ]
    bertscores = [
        float(record.component_metrics["bertscore_f1"])
        for record in records
        if "bertscore_f1" in record.component_metrics
    ]
    return TaskSummary(
        model_name=model_name,
        task_name=task_name,
        protocol_id=protocol_id,
        num_samples=total_num_samples,
        task_accuracy=(
            sum(record.task_pass for record in records)
            / len(records)
            if records
            else None
        ),
        num_scored_samples=len(records),
        num_pending_samples=max(total_num_samples - len(records), 0),
        judge_pass_rate=_mean_or_none(judge_passes),
        bbox_pass_rate=_mean_or_none(bbox_passes),
        tiou_pass_rate=_mean_or_none(tiou_passes),
        tracking_mean_iou_mean=_mean_or_none(tracking_mean_ious),
        tracking_pass_rate_mean=_mean_or_none(tracking_pass_rates),
        mean_bertscore_f1=_mean_or_none(bertscores),
        median_bertscore_f1=statistics.median(bertscores) if bertscores else None,
    )
