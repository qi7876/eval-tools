"""Strict validation for structured benchmark predictions."""

from __future__ import annotations

from typing import Any

from .constants import (
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_SINGLE,
    TASK_STG,
    TEXT_ONLY_TASKS,
)
from .schema import PreparedSample, StructuredPredictionResult


def _coerce_text_field(payload: dict[str, Any], errors: list[str]) -> str:
    if "text" not in payload:
        errors.append("missing text field")
        return ""
    return str(payload["text"]).strip()


def _coerce_box(value: Any, field_name: str, errors: list[str]) -> list[float]:
    if not isinstance(value, list):
        errors.append(f"{field_name} must be a list")
        return []
    if len(value) == 0:
        return []
    if len(value) != 4:
        errors.append(f"{field_name} must be empty or contain exactly 4 values")
        return []
    return [float(item) for item in value]


def _scoreboard_bbox_or_sentinel(value: Any, warnings: list[str]) -> list[float]:
    sentinel = [-1.0, -1.0, -1.0, -1.0]
    if value is None:
        warnings.append("bbox missing; using sentinel bbox")
        return sentinel
    if not isinstance(value, list):
        warnings.append("bbox invalid; using sentinel bbox")
        return sentinel
    if len(value) != 4:
        warnings.append("bbox invalid; using sentinel bbox")
        return sentinel
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        warnings.append("bbox invalid; using sentinel bbox")
        return sentinel


def _required_object_labels(prepared_sample: PreparedSample, errors: list[str]) -> list[str]:
    gt_objects = prepared_sample.reference_payload.get("objects")
    if not isinstance(gt_objects, list):
        errors.append("reference objects must be a list")
        return []
    labels: list[str] = []
    for index, gt_object in enumerate(gt_objects):
        if not isinstance(gt_object, dict):
            errors.append(f"reference objects[{index}] must be an object")
            continue
        label = str(gt_object.get("label", "")).strip()
        if not label:
            errors.append(f"reference objects[{index}].label must be non-empty")
            continue
        labels.append(label)
    return labels


def _coerce_labeled_objects(
    value: Any,
    prepared_sample: PreparedSample,
    errors: list[str],
) -> list[dict[str, Any]]:
    required_labels = _required_object_labels(prepared_sample, errors)
    if not isinstance(value, list):
        errors.append("objects must be a list")
        return []
    predicted_by_label: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            errors.append(f"objects[{index}] must be an object")
            continue
        if {"label", "bbox"} - set(row):
            errors.append(f"objects[{index}] must contain label and bbox")
            continue
        label = str(row["label"]).strip()
        if not label:
            errors.append(f"objects[{index}].label must be non-empty")
            continue
        if label in predicted_by_label:
            errors.append(f"duplicate object label: {label}")
            continue
        bbox = _coerce_box(row["bbox"], f"objects[{index}].bbox", errors)
        if len(bbox) != 4:
            continue
        predicted_by_label[label] = {"label": label, "bbox": bbox}

    required_label_set = set(required_labels)
    for label in sorted(predicted_by_label):
        if label not in required_label_set:
            errors.append(f"unexpected object label: {label}")
    for label in required_labels:
        if label not in predicted_by_label:
            errors.append(f"missing object label: {label}")
    if errors:
        return []
    return [predicted_by_label[label] for label in required_labels]


def _coerce_segments(
    value: Any,
    num_sampled_frames: int,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        errors.append("segments must be a list")
        return []
    segments: list[dict[str, Any]] = []
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            errors.append(f"segments[{index}] must be an object")
            continue
        if {"start_sampled", "end_sampled", "text"} - set(segment):
            errors.append(
                f"segments[{index}] must contain start_sampled, end_sampled, and text"
            )
            continue
        start = int(segment["start_sampled"])
        end = int(segment["end_sampled"])
        if start < 0 or end < 0 or start >= num_sampled_frames or end >= num_sampled_frames:
            errors.append(f"segments[{index}] index out of range: {start}-{end}")
            continue
        if start > end:
            errors.append(f"segments[{index}] start > end: {start}-{end}")
            continue
        segments.append(
            {
                "start_sampled": start,
                "end_sampled": end,
                "text": str(segment["text"]).strip(),
            }
        )
    return segments


def _coerce_tracking(
    value: Any,
    num_sampled_frames: int,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        errors.append("tracking must be a list")
        return []
    tracking: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            errors.append(f"tracking[{index}] must be an object")
            continue
        if {"frame_sampled", "bbox_mot"} - set(row):
            errors.append(f"tracking[{index}] must contain frame_sampled and bbox_mot")
            continue
        frame_sampled = int(row["frame_sampled"])
        if frame_sampled < 0 or frame_sampled >= num_sampled_frames:
            errors.append(f"tracking[{index}] frame_sampled out of range: {frame_sampled}")
            continue
        bbox_mot = _coerce_box(row["bbox_mot"], f"tracking[{index}].bbox_mot", errors)
        if len(bbox_mot) != 4:
            continue
        tracking.append(
            {
                "frame_sampled": frame_sampled,
                "bbox_mot": bbox_mot,
            }
        )
    return tracking


def _coerce_time_window(
    value: Any,
    num_sampled_frames: int,
    errors: list[str],
) -> list[int]:
    if not isinstance(value, list):
        errors.append("time_window_sampled must be a list")
        return []
    if len(value) == 0:
        return []
    if len(value) != 2:
        errors.append("time_window_sampled must be empty or contain exactly 2 values")
        return []
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end < 0 or start >= num_sampled_frames or end >= num_sampled_frames:
        errors.append(f"time_window_sampled out of range: {start}-{end}")
        return []
    if start > end:
        errors.append(f"time_window_sampled start > end: {start}-{end}")
        return []
    return [start, end]


def validate_structured_prediction(
    prepared_sample: PreparedSample,
    raw_output: str,
    structured_prediction: dict[str, Any],
    *,
    structurer_raw_response: str | None = None,
    oracle_upstream: bool = False,
) -> StructuredPredictionResult:
    errors: list[str] = []
    warnings: list[str] = []
    num_sampled_frames = len(prepared_sample.sampled_frames_original)
    task_name = prepared_sample.task_name

    if task_name in TEXT_ONLY_TASKS:
        validated = {"text": _coerce_text_field(structured_prediction, errors)}
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    if task_name == TASK_SCOREBOARD_SINGLE:
        validated = {
            "text": _coerce_text_field(structured_prediction, errors),
            "bbox": _scoreboard_bbox_or_sentinel(structured_prediction.get("bbox"), warnings),
        }
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    if task_name == TASK_OBJECTS_SPATIAL:
        validated = {
            "text": _coerce_text_field(structured_prediction, errors),
            "objects": _coerce_labeled_objects(
                structured_prediction.get("objects"),
                prepared_sample,
                errors,
            ),
        }
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    if task_name == TASK_CONTINUOUS_EVENTS:
        validated = {
            "segments": _coerce_segments(
                structured_prediction.get("segments"),
                num_sampled_frames,
                errors,
            )
        }
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    if task_name == TASK_CONTINUOUS_ACTIONS:
        if oracle_upstream:
            validated = {
                "segments": _coerce_segments(
                    structured_prediction.get("segments"),
                    num_sampled_frames,
                    errors,
                )
            }
            return StructuredPredictionResult(
                task_name=task_name,
                raw_output=raw_output,
                structured_prediction=(validated if not errors else None),
                errors=errors,
                warnings=warnings,
                structurer_raw_response=structurer_raw_response,
            )
        validated = {
            "segments": _coerce_segments(
                structured_prediction.get("segments"),
                num_sampled_frames,
                errors,
            ),
            "tracking": _coerce_tracking(
                structured_prediction.get("tracking"),
                num_sampled_frames,
                errors,
            ),
        }
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    if task_name == TASK_STG:
        if oracle_upstream:
            validated = {
                "time_window_sampled": _coerce_time_window(
                    structured_prediction.get("time_window_sampled"),
                    num_sampled_frames,
                    errors,
                ),
            }
            return StructuredPredictionResult(
                task_name=task_name,
                raw_output=raw_output,
                structured_prediction=(validated if not errors else None),
                errors=errors,
                warnings=warnings,
                structurer_raw_response=structurer_raw_response,
            )
        validated = {
            "time_window_sampled": _coerce_time_window(
                structured_prediction.get("time_window_sampled"),
                num_sampled_frames,
                errors,
            ),
            "tracking": _coerce_tracking(
                structured_prediction.get("tracking"),
                num_sampled_frames,
                errors,
            ),
        }
        return StructuredPredictionResult(
            task_name=task_name,
            raw_output=raw_output,
            structured_prediction=(validated if not errors else None),
            errors=errors,
            warnings=warnings,
            structurer_raw_response=structurer_raw_response,
        )

    errors.append(f"unsupported task for structured validation: {task_name}")
    return StructuredPredictionResult(
        task_name=task_name,
        raw_output=raw_output,
        structured_prediction=None,
        errors=errors,
        warnings=warnings,
        structurer_raw_response=structurer_raw_response,
    )
