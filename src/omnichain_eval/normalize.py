"""Canonical prediction normalization."""

from __future__ import annotations

from typing import Any

from .constants import (
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_SINGLE,
    TASK_STG,
    TEXT_ONLY_TASKS,
)
from .schema import NormalizationResult
from .utils import extract_json_object


def _coerce_raw(raw_output: Any) -> Any:
    if isinstance(raw_output, (dict, list)):
        return raw_output
    if isinstance(raw_output, str):
        try:
            return extract_json_object(raw_output)
        except ValueError:
            return raw_output.strip()
    return raw_output


def _coerce_text(parsed: Any) -> str | None:
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        value = parsed.get("text")
        if value is None:
            return None
        return str(value).strip()
    return None


def _coerce_bbox(parsed: Any, *keys: str) -> list[float] | None:
    if not isinstance(parsed, dict):
        return None
    value = None
    for key in keys:
        if key in parsed:
            value = parsed[key]
            break
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        return None
    return [float(item) for item in value]


def _coerce_segments(parsed: Any) -> list[dict[str, Any]] | None:
    if not isinstance(parsed, dict):
        return None
    raw_segments = parsed.get("segments")
    if not isinstance(raw_segments, list):
        return None
    segments: list[dict[str, Any]] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            return None
        start = segment.get("start_sampled", segment.get("start"))
        end = segment.get("end_sampled", segment.get("end"))
        text = segment.get("text")
        if start is None or end is None or text is None:
            return None
        segments.append(
            {
                "start_sampled": int(start),
                "end_sampled": int(end),
                "text": str(text).strip(),
            }
        )
    return segments


def _coerce_tracking(parsed: Any) -> list[dict[str, Any]] | None:
    if not isinstance(parsed, dict):
        return None
    raw_tracking = parsed.get("tracking")
    if raw_tracking is None:
        return []
    if not isinstance(raw_tracking, list):
        return None
    tracking: list[dict[str, Any]] = []
    for row in raw_tracking:
        if not isinstance(row, dict):
            return None
        frame_sampled = row.get("frame_sampled", row.get("sampled_frame", row.get("frame")))
        bbox = row.get("bbox_mot", row.get("bbox"))
        if frame_sampled is None or bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
            return None
        tracking.append(
            {
                "frame_sampled": int(frame_sampled),
                "bbox_mot": [float(value) for value in bbox],
            }
        )
    return tracking


def normalize_prediction(task_name: str, raw_output: Any) -> NormalizationResult:
    parsed = _coerce_raw(raw_output)
    errors: list[str] = []
    warnings: list[str] = []

    if task_name in TEXT_ONLY_TASKS:
        text = _coerce_text(parsed)
        if not text:
            errors.append("missing text field")
            normalized = None
        else:
            normalized = {"text": text}
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    if task_name == TASK_SCOREBOARD_SINGLE:
        text = _coerce_text(parsed)
        bbox = _coerce_bbox(parsed, "bbox", "bounding_box", "box")
        if not text:
            errors.append("missing text field")
        if bbox is None:
            errors.append("missing bbox field")
        normalized = {"text": text, "bbox": bbox} if not errors else None
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    if task_name == TASK_OBJECTS_SPATIAL:
        text = _coerce_text(parsed)
        bbox_a = _coerce_bbox(parsed, "bbox_a", "box_a")
        bbox_b = _coerce_bbox(parsed, "bbox_b", "box_b")
        if bbox_a is None and isinstance(parsed, dict) and isinstance(parsed.get("boxes"), list):
            boxes = parsed["boxes"]
            if len(boxes) == 2 and all(isinstance(box, list) and len(box) == 4 for box in boxes):
                bbox_a = [float(value) for value in boxes[0]]
                bbox_b = [float(value) for value in boxes[1]]
        if not text:
            errors.append("missing text field")
        if bbox_a is None:
            errors.append("missing bbox_a field")
        if bbox_b is None:
            errors.append("missing bbox_b field")
        normalized = (
            {"text": text, "bbox_a": bbox_a, "bbox_b": bbox_b} if not errors else None
        )
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    if task_name in {TASK_CONTINUOUS_EVENTS, TASK_COMMENTARY}:
        segments = _coerce_segments(parsed)
        if segments is None:
            errors.append("missing segments field")
            normalized = None
        else:
            normalized = {"segments": segments}
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    if task_name == TASK_CONTINUOUS_ACTIONS:
        segments = _coerce_segments(parsed)
        tracking = _coerce_tracking(parsed)
        if segments is None:
            errors.append("missing segments field")
        if tracking is None:
            errors.append("invalid tracking field")
        normalized = (
            {"segments": segments, "tracking": tracking if tracking is not None else []}
            if not errors
            else None
        )
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    if task_name == TASK_STG:
        if not isinstance(parsed, dict):
            errors.append("prediction must be a JSON object")
            return NormalizationResult(task_name, raw_output, None, errors, warnings)
        window = parsed.get("time_window_sampled", parsed.get("time_window", parsed.get("window")))
        tracking = _coerce_tracking(parsed)
        if not isinstance(window, list) or len(window) != 2:
            errors.append("missing time_window_sampled field")
        if tracking is None:
            errors.append("invalid tracking field")
        normalized = (
            {
                "time_window_sampled": [int(window[0]), int(window[1])],
                "tracking": tracking if tracking is not None else [],
            }
            if not errors
            else None
        )
        return NormalizationResult(task_name, raw_output, normalized, errors, warnings)

    errors.append(f"unsupported task for normalization: {task_name}")
    return NormalizationResult(task_name, raw_output, None, errors, warnings)
