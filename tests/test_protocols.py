from pathlib import Path

from omnichain_eval.protocols import (
    ALL_PROTOCOLS,
    MAIN_PROTOCOL,
    get_protocol,
    original_interval_to_sampled_interval,
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


def test_only_main_protocol_is_registered():
    assert set(ALL_PROTOCOLS) == {"main"}


def test_get_protocol_rejects_removed_experiment_d_protocol():
    try:
        get_protocol("expd_fps_32s_4fps")
    except KeyError as exc:
        assert "main-only" in str(exc)
    else:
        raise AssertionError("expected KeyError for removed Experiment D protocol")


def test_original_interval_to_sampled_interval():
    sampled = [100, 105, 110, 115, 120]
    assert original_interval_to_sampled_interval(106, 119, sampled) == [2, 3]
