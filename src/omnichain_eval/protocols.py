"""Sampling protocol definitions and helpers."""

from __future__ import annotations

from bisect import bisect_left
from typing import Iterable

from .constants import (
    EXPERIMENT_D_TASKS,
    MAIN_FRAME_BUDGET,
    SHORT_WINDOW_FRAMES,
    SINGLE_FRAME_TASKS,
    STG_EXPANSION_FRAMES,
    TASK_STG,
    VIDEO_FPS,
)
from .schema import ProtocolSpec, SampleRecord
from .utils import clip_index


MAIN_PROTOCOL = ProtocolSpec(
    protocol_id="main",
    description="Main fixed-budget benchmark protocol.",
    frame_budget=MAIN_FRAME_BUDGET,
    supports_stg=True,
    include_single_frame_tasks=True,
    strategy="main",
)

EXPERIMENT_D_PROTOCOLS = {
    "expd_window_16s_2fps": ProtocolSpec(
        protocol_id="expd_window_16s_2fps",
        description="Experiment D window ablation: 16s @ 2fps.",
        frame_budget=32,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=16,
        target_fps=2.0,
        strategy="recent_history_fixed_fps",
    ),
    "expd_window_32s_2fps": ProtocolSpec(
        protocol_id="expd_window_32s_2fps",
        description="Experiment D window ablation: 32s @ 2fps.",
        frame_budget=64,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=32,
        target_fps=2.0,
        strategy="recent_history_fixed_fps",
    ),
    "expd_window_64s_2fps": ProtocolSpec(
        protocol_id="expd_window_64s_2fps",
        description="Experiment D window ablation: 64s @ 2fps.",
        frame_budget=128,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=64,
        target_fps=2.0,
        strategy="recent_history_fixed_fps",
    ),
    "expd_fps_32s_1fps": ProtocolSpec(
        protocol_id="expd_fps_32s_1fps",
        description="Experiment D fps ablation: 32s @ 1fps.",
        frame_budget=32,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=32,
        target_fps=1.0,
        strategy="recent_history_fixed_fps",
    ),
    "expd_fps_32s_2fps": ProtocolSpec(
        protocol_id="expd_fps_32s_2fps",
        description="Experiment D fps ablation: 32s @ 2fps.",
        frame_budget=64,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=32,
        target_fps=2.0,
        strategy="recent_history_fixed_fps",
    ),
    "expd_fps_32s_4fps": ProtocolSpec(
        protocol_id="expd_fps_32s_4fps",
        description="Experiment D fps ablation: 32s @ 4fps.",
        frame_budget=128,
        supports_stg=False,
        include_single_frame_tasks=False,
        history_seconds=32,
        target_fps=4.0,
        strategy="recent_history_fixed_fps",
    ),
}

ALL_PROTOCOLS = {MAIN_PROTOCOL.protocol_id: MAIN_PROTOCOL, **EXPERIMENT_D_PROTOCOLS}


def get_protocol(protocol_id: str) -> ProtocolSpec:
    try:
        return ALL_PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise KeyError(f"unknown protocol_id: {protocol_id}") from exc


def protocol_supports_task(protocol: ProtocolSpec, task_name: str) -> bool:
    if protocol.protocol_id == MAIN_PROTOCOL.protocol_id:
        return True
    if task_name == TASK_STG:
        return protocol.supports_stg
    if task_name in SINGLE_FRAME_TASKS:
        return protocol.include_single_frame_tasks
    return task_name in EXPERIMENT_D_TASKS


def _remove_duplicates(values: Iterable[int]) -> list[int]:
    deduped: list[int] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


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


def sample_recent_history(
    anchor_end: int,
    total_frames: int,
    *,
    history_seconds: int,
    target_fps: float,
    budget: int,
) -> list[int]:
    span_frames = history_seconds * VIDEO_FPS
    visible_start = max(0, anchor_end - span_frames + 1)
    visible_end = min(anchor_end, total_frames - 1)
    stride = VIDEO_FPS / target_fps
    candidates: list[int] = []
    for index in range(budget):
        float_position = visible_end - stride * index
        rounded = clip_index(round(float_position), total_frames)
        if rounded < visible_start:
            break
        candidates.append(rounded)
    return list(reversed(_remove_duplicates(candidates)))


def sample_frames_for_sample(sample: SampleRecord, protocol: ProtocolSpec) -> list[int]:
    total_frames = sample.video_metadata.total_frames
    if sample.task_name in SINGLE_FRAME_TASKS:
        if sample.timestamp_frame is None:
            raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
        return [sample.timestamp_frame]
    if sample.task_name == TASK_STG:
        if not protocol.supports_stg:
            raise ValueError(f"{protocol.protocol_id} does not support STG")
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
    q_start, q_end = sample.q_window
    if protocol.strategy == "main":
        interval_length = q_end - q_start + 1
        if interval_length <= SHORT_WINDOW_FRAMES:
            visible_start = max(0, q_end - SHORT_WINDOW_FRAMES + 1)
            candidates = list(range(q_end, visible_start - 1, -5))
            return list(reversed(candidates))
        return uniform_sample_closed_interval(q_start, q_end, protocol.frame_budget)
    if protocol.strategy == "recent_history_fixed_fps":
        if protocol.history_seconds is None or protocol.target_fps is None:
            raise ValueError(f"protocol {protocol.protocol_id} is missing history/fps parameters")
        return sample_recent_history(
            q_end,
            total_frames,
            history_seconds=protocol.history_seconds,
            target_fps=protocol.target_fps,
            budget=protocol.frame_budget,
        )
    raise ValueError(f"unsupported sampling strategy: {protocol.strategy}")


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
