from pathlib import Path
from textwrap import dedent

import pytest

from omnichain_eval.protocols import (
    ALL_PROTOCOLS,
    MAIN_PROTOCOL,
    get_protocol,
    original_interval_to_sampled_interval,
    resolve_protocol,
    sample_frames_for_sample,
)
from omnichain_eval.schema import SampleRecord, VideoMetadata


def build_sample(task_name: str, **kwargs) -> SampleRecord:
    total_frames = kwargs.pop("total_frames", 1000)
    return SampleRecord(
        sample_id="video#1",
        annotation_id="1",
        video_key="video",
        task_name=task_name,
        task_level="Understanding",
        question_text="test",
        source_annotation_path=Path("video.json"),
        source_video_path=Path("video.mp4"),
        video_metadata=VideoMetadata(10.0, 10, total_frames, (1920, 1080)),
        raw_annotation={},
        reference_payload={"text": "answer"},
        **kwargs,
    )


def _write_protocol_module(tmp_path: Path) -> str:
    module_name = "custom_protocols_protocols"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from omnichain_eval.constants import SINGLE_FRAME_TASKS, TASK_STG
            from omnichain_eval.protocols import BaseProtocol, uniform_sample_closed_interval
            from omnichain_eval.utils import clip_index


            def _clip_closed_interval(start: int, end: int, total_frames: int) -> tuple[int, int]:
                clipped_start = clip_index(start, total_frames)
                clipped_end = clip_index(end, total_frames)
                if clipped_start > clipped_end:
                    clipped_start, clipped_end = clipped_end, clipped_start
                return clipped_start, clipped_end


            class EightFrameUniformProtocol(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Eight-frame uniform sampling."

                def sample_frames(self, sample):
                    total_frames = sample.video_metadata.total_frames
                    if sample.task_name in SINGLE_FRAME_TASKS:
                        if sample.timestamp_frame is None:
                            raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
                        return [sample.timestamp_frame]
                    if sample.task_name == TASK_STG:
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

                def to_manifest_dict(self):
                    return {
                        **super().to_manifest_dict(),
                        "sampling": "uniform",
                        "frame_budget": 8,
                    }


            class EightFrameUniformProtocolAlt(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Alternate eight-frame uniform sampling."

                def sample_frames(self, sample):
                    total_frames = sample.video_metadata.total_frames
                    if sample.task_name in SINGLE_FRAME_TASKS:
                        if sample.timestamp_frame is None:
                            raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
                        return [sample.timestamp_frame]
                    if sample.task_name == TASK_STG:
                        if sample.a_window is None:
                            raise ValueError(f"{sample.sample_id} is missing A_window_frame")
                        start, end = _clip_closed_interval(
                            sample.a_window[0],
                            sample.a_window[1],
                            total_frames,
                        )
                        return uniform_sample_closed_interval(start, end, 4)
                    if sample.q_window is None:
                        raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
                    start, end = _clip_closed_interval(
                        sample.q_window[0],
                        sample.q_window[1],
                        total_frames,
                    )
                    return uniform_sample_closed_interval(start, end, 4)

                def to_manifest_dict(self):
                    return {
                        **super().to_manifest_dict(),
                        "sampling": "uniform",
                        "frame_budget": 4,
                    }
            """
        ),
        encoding="utf-8",
    )
    return module_name


def test_main_short_window_sampling():
    sample = build_sample("Scoreboard_Multiple", q_window=(100, 150))
    frames = sample_frames_for_sample(sample, MAIN_PROTOCOL)
    assert frames[0] == 0
    assert frames[-1] == 150
    assert len(frames) == 31


def test_main_long_window_sampling_has_budget():
    sample = build_sample("Score_Prediction", q_window=(0, 999))
    frames = sample_frames_for_sample(sample, MAIN_PROTOCOL)
    assert len(frames) == 64
    assert frames[0] == 0
    assert frames[-1] == 999


def test_main_long_window_sampling_clips_query_end_to_last_frame():
    sample = build_sample("Score_Prediction", q_window=(0, 1000))
    frames = sample_frames_for_sample(sample, MAIN_PROTOCOL)
    assert len(frames) == 64
    assert frames[0] == 0
    assert frames[-1] == 999
    assert 1000 not in frames


def test_main_short_window_sampling_clips_query_end_to_last_frame():
    sample = build_sample("Scoreboard_Multiple", q_window=(900, 1000))
    frames = sample_frames_for_sample(sample, MAIN_PROTOCOL)
    assert frames[-1] == 999
    assert 1000 not in frames


def test_only_main_protocol_is_built_in():
    assert set(ALL_PROTOCOLS) == {"main"}


def test_get_protocol_rejects_unknown_built_in_protocol():
    with pytest.raises(KeyError, match="unknown built-in protocol_id"):
        get_protocol("expd_fps_32s_4fps")


def test_resolve_protocol_accepts_python_spec(tmp_path, monkeypatch):
    module_name = _write_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)

    protocol = resolve_protocol(f"{module_name}:EightFrameUniformProtocol")
    sample = build_sample("Score_Prediction", q_window=(100, 150))

    assert protocol.protocol_id == "native_uniform_8"
    assert sample_frames_for_sample(sample, protocol) == [100, 107, 114, 121, 129, 136, 143, 150]


def test_original_interval_to_sampled_interval():
    sampled = [100, 105, 110, 115, 120]
    assert original_interval_to_sampled_interval(106, 119, sampled) == [2, 3]
