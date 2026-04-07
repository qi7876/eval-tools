"""Importable example protocols for config templates and documentation."""

from __future__ import annotations

from .constants import SINGLE_FRAME_TASKS
from .protocols import BaseProtocol, uniform_sample_closed_interval
from .utils import clip_index


def _clip_closed_interval(start: int, end: int, total_frames: int) -> tuple[int, int]:
    clipped_start = clip_index(start, total_frames)
    clipped_end = clip_index(end, total_frames)
    if clipped_start > clipped_end:
        clipped_start, clipped_end = clipped_end, clipped_start
    return clipped_start, clipped_end


class ExampleEightFrameUniformProtocol(BaseProtocol):
    """Example native-sampling protocol used by docs and example TOMLs."""

    @property
    def protocol_id(self) -> str:
        return "example_uniform_8"

    @property
    def description(self) -> str:
        return "Example protocol: eight-frame uniform sampling."

    def sample_frames(self, sample) -> list[int]:
        total_frames = sample.video_metadata.total_frames
        if sample.task_name in SINGLE_FRAME_TASKS:
            if sample.timestamp_frame is None:
                raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
            return [sample.timestamp_frame]
        if sample.task_name == "Spatial_Temporal_Grounding":
            if sample.a_window is None:
                raise ValueError(f"{sample.sample_id} is missing A_window_frame")
            start, end = _clip_closed_interval(
                sample.a_window[0],
                sample.a_window[1],
                total_frames,
            )
            return uniform_sample_closed_interval(start, end, 8)
        if sample.q_window is None:
            raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
        start, end = _clip_closed_interval(
            sample.q_window[0],
            sample.q_window[1],
            total_frames,
        )
        return uniform_sample_closed_interval(start, end, 8)

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            **super().to_manifest_dict(),
            "sampling": "uniform",
            "frame_budget": 8,
        }
