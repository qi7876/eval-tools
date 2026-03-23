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
            "bbox": _coerce_box(structured_prediction.get("bbox"), "bbox", errors),
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
            "bbox_a": _coerce_box(structured_prediction.get("bbox_a"), "bbox_a", errors),
            "bbox_b": _coerce_box(structured_prediction.get("bbox_b"), "bbox_b", errors),
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
