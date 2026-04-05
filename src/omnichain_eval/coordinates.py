"""Coordinate helpers for normalized 0..1000 boxes."""

from __future__ import annotations

from .constants import NORMALIZED_COORDINATE_MAX


def _validate_frame_size(frame_width: int, frame_height: int) -> None:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame dimensions must be positive, got width={frame_width}, height={frame_height}"
        )


def normalize_corner_box_from_pixels(
    box: list[float],
    *,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    _validate_frame_size(frame_width, frame_height)
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        x1 * NORMALIZED_COORDINATE_MAX / frame_width,
        y1 * NORMALIZED_COORDINATE_MAX / frame_height,
        x2 * NORMALIZED_COORDINATE_MAX / frame_width,
        y2 * NORMALIZED_COORDINATE_MAX / frame_height,
    ]


def normalize_mot_box_from_pixels(
    box: list[float],
    *,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    _validate_frame_size(frame_width, frame_height)
    left, top, width, height = [float(value) for value in box]
    return [
        left * NORMALIZED_COORDINATE_MAX / frame_width,
        top * NORMALIZED_COORDINATE_MAX / frame_height,
        width * NORMALIZED_COORDINATE_MAX / frame_width,
        height * NORMALIZED_COORDINATE_MAX / frame_height,
    ]


def denormalize_mot_box_to_pixel_corners(
    box: list[float],
    *,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    _validate_frame_size(frame_width, frame_height)
    left, top, width, height = [float(value) for value in box]
    x1 = left * frame_width / NORMALIZED_COORDINATE_MAX
    y1 = top * frame_height / NORMALIZED_COORDINATE_MAX
    x2 = (left + width) * frame_width / NORMALIZED_COORDINATE_MAX
    y2 = (top + height) * frame_height / NORMALIZED_COORDINATE_MAX
    max_x = frame_width - 1
    max_y = frame_height - 1
    return [
        max(0, min(max_x, round(x1))),
        max(0, min(max_y, round(y1))),
        max(0, min(max_x, round(x2))),
        max(0, min(max_y, round(y2))),
    ]
