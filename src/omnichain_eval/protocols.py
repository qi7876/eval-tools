"""Sampling protocol definitions and helpers."""

from __future__ import annotations

from bisect import bisect_left
from typing import Iterable

from .constants import (
    MAIN_FRAME_BUDGET,
    SHORT_WINDOW_FRAMES,
    SINGLE_FRAME_TASKS,
    STG_EXPANSION_FRAMES,
    TASK_STG,
)
from .schema import ProtocolSpec, SampleRecord
from .utils import clip_index


MAIN_PROTOCOL = ProtocolSpec(
    protocol_id="main",
    description="Main fixed-budget benchmark protocol.",
    frame_budget=MAIN_FRAME_BUDGET,
)

ALL_PROTOCOLS = {MAIN_PROTOCOL.protocol_id: MAIN_PROTOCOL}


def get_protocol(protocol_id: str) -> ProtocolSpec:
    try:
        return ALL_PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown protocol_id: {protocol_id}; current prepared-data support is main-only"
        ) from exc


def protocol_supports_task(protocol: ProtocolSpec, task_name: str) -> bool:
    del task_name
    return protocol.protocol_id == MAIN_PROTOCOL.protocol_id


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


def sample_frames_for_sample(sample: SampleRecord, protocol: ProtocolSpec) -> list[int]:
    if protocol.protocol_id != MAIN_PROTOCOL.protocol_id:
        raise ValueError(
            f"unsupported protocol_id {protocol.protocol_id!r}; current prepared-data support is main-only"
        )
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
            protocol.frame_budget,
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
    return uniform_sample_closed_interval(q_start, q_end, protocol.frame_budget)


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
