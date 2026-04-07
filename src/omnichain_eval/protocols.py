"""Sampling protocol definitions and helpers."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Any, Iterable

from .constants import (
    MAIN_FRAME_BUDGET,
    SHORT_WINDOW_FRAMES,
    SINGLE_FRAME_TASKS,
    STG_EXPANSION_FRAMES,
    TASK_STG,
)
from .schema import SampleRecord
from .utils import clip_index

def _remove_duplicates(values: Iterable[int]) -> list[int]:
    deduped: list[int] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def _clip_closed_interval(start: int, end: int, total_frames: int) -> tuple[int, int]:
    clipped_start = clip_index(start, total_frames)
    clipped_end = clip_index(end, total_frames)
    if clipped_start > clipped_end:
        clipped_start, clipped_end = clipped_end, clipped_start
    return clipped_start, clipped_end


def uniform_sample_closed_interval(start: int, end: int, budget: int) -> list[int]:
    if budget <= 1 or start == end:
        return [start]
    samples: list[int] = []
    for index in range(budget):
        position = start + (index / (budget - 1)) * (end - start)
        candidate = round(position)
        candidate = min(max(candidate, start), end)
        samples.append(candidate)
    return _remove_duplicates(samples)


class BaseProtocol(ABC):
    """Runtime sampling protocol."""

    @property
    @abstractmethod
    def protocol_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    def supports_task(self, task_name: str) -> bool:
        del task_name
        return True

    @abstractmethod
    def sample_frames(self, sample: SampleRecord) -> list[int]:
        raise NotImplementedError

    def project_question_window(
        self,
        sample: SampleRecord,
        sampled_frames_original: list[int],
    ) -> tuple[int, int] | None:
        if sample.q_window is None:
            return None
        return tuple(
            original_interval_to_sampled_interval(
                sample.q_window[0],
                sample.q_window[1],
                sampled_frames_original,
            )
        )

    def project_answer_window(
        self,
        sample: SampleRecord,
        sampled_frames_original: list[int],
    ) -> tuple[int, int] | None:
        if sample.a_window is None:
            return None
        return tuple(
            original_interval_to_sampled_interval(
                sample.a_window[0],
                sample.a_window[1],
                sampled_frames_original,
            )
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "description": self.description,
        }


class MainProtocol(BaseProtocol):
    @property
    def protocol_id(self) -> str:
        return "main"

    @property
    def description(self) -> str:
        return "Main fixed-budget benchmark protocol."

    @property
    def frame_budget(self) -> int:
        return MAIN_FRAME_BUDGET

    def sample_frames(self, sample: SampleRecord) -> list[int]:
        total_frames = sample.video_metadata.total_frames
        if sample.task_name in SINGLE_FRAME_TASKS:
            if sample.timestamp_frame is None:
                raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
            return [sample.timestamp_frame]
        if sample.task_name == TASK_STG:
            if sample.a_window is None:
                raise ValueError(f"{sample.sample_id} is missing A_window_frame")
            answer_start, answer_end = sample.a_window
            visible_start = clip_index(answer_start - STG_EXPANSION_FRAMES, total_frames)
            visible_end = clip_index(answer_end + STG_EXPANSION_FRAMES, total_frames)
            if visible_start > visible_end:
                visible_start, visible_end = visible_end, visible_start
            return uniform_sample_closed_interval(
                visible_start,
                visible_end,
                self.frame_budget,
            )
        if sample.q_window is None:
            raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
        q_start, q_end = _clip_closed_interval(
            sample.q_window[0],
            sample.q_window[1],
            total_frames,
        )
        interval_length = q_end - q_start + 1
        if interval_length <= SHORT_WINDOW_FRAMES:
            visible_start = max(0, q_end - SHORT_WINDOW_FRAMES + 1)
            candidates = list(range(q_end, visible_start - 1, -5))
            return list(reversed(candidates))
        return uniform_sample_closed_interval(q_start, q_end, self.frame_budget)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            **super().to_manifest_dict(),
            "frame_budget": self.frame_budget,
            "sampling": "main_fixed_budget",
        }


MAIN_PROTOCOL = MainProtocol()
ALL_PROTOCOLS = {MAIN_PROTOCOL.protocol_id: MAIN_PROTOCOL}


def get_protocol(protocol_id: str) -> BaseProtocol:
    try:
        return ALL_PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise KeyError(f"unknown built-in protocol_id: {protocol_id!r}") from exc


def resolve_protocol(spec: str) -> BaseProtocol:
    if spec in ALL_PROTOCOLS:
        return ALL_PROTOCOLS[spec]
    if ":" not in spec:
        raise ValueError("protocol spec must be a built-in id or 'module.path:ClassName'")
    module_name, class_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    protocol_cls = getattr(module, class_name)
    protocol = protocol_cls()
    if not isinstance(protocol, BaseProtocol):
        raise TypeError(f"{spec} did not resolve to a BaseProtocol instance")
    if not protocol.protocol_id:
        raise ValueError(f"{spec} produced an empty protocol_id")
    return protocol


def protocol_supports_task(protocol: BaseProtocol, task_name: str) -> bool:
    return protocol.supports_task(task_name)


def sample_frames_for_sample(sample: SampleRecord, protocol: BaseProtocol) -> list[int]:
    return protocol.sample_frames(sample)


def sampled_to_original_mapping(sampled_frames_original: list[int]) -> dict[int, int]:
    return {index: frame for index, frame in enumerate(sampled_frames_original)}


def original_interval_to_sampled_interval(
    start_frame: int,
    end_frame: int,
    sampled_frames_original: list[int],
) -> list[int]:
    inside = [
        index
        for index, frame_index in enumerate(sampled_frames_original)
        if start_frame <= frame_index <= end_frame
    ]
    if inside:
        return [inside[0], inside[-1]]
    insertion_start = bisect_left(sampled_frames_original, start_frame)
    insertion_end = bisect_left(sampled_frames_original, end_frame)
    candidate_indices = [
        min(max(insertion_start, 0), len(sampled_frames_original) - 1),
        min(max(insertion_end, 0), len(sampled_frames_original) - 1),
    ]
    candidate_indices.sort()
    return [candidate_indices[0], candidate_indices[-1]]
