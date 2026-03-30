from omnichain_eval.judge import StaticJudgeClient
from omnichain_eval.metrics import evaluate_sample
from omnichain_eval.normalize import validate_structured_prediction
from omnichain_eval.schema import PreparedSample, StructuredPredictionRecord


def _sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/sample#1",
        annotation_id="1",
        video_key="video/sample",
        task_name="Scoreboard_Single",
        task_level="independent",
        protocol_id="main",
        question_text="What is the current score?",
        sampled_frames_original=[12],
        sampled_to_original={0: 12},
        frame_files=["frames/0000.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "text": "1-0",
            "bbox": [10.0, 20.0, 110.0, 80.0],
        },
    )


def _record(structured_prediction: dict | None, *, raw_output: str = "{}") -> StructuredPredictionRecord:
    return StructuredPredictionRecord(
        sample_id="video/sample#1",
        task_name="Scoreboard_Single",
        video_key="video/sample",
        protocol_id="main",
        raw_output=raw_output,
        structured_prediction=structured_prediction,
        structuring_errors=[],
        structuring_warnings=[],
    )


def test_validate_scoreboard_single_missing_bbox_uses_sentinel():
    result = validate_structured_prediction(
        _sample(),
        raw_output='{"text":"1-0"}',
        structured_prediction={"text": "1-0"},
    )

    assert result.errors == []
    assert result.warnings == ["bbox missing; using sentinel bbox"]
    assert result.structured_prediction == {
        "text": "1-0",
        "bbox": [-1.0, -1.0, -1.0, -1.0],
    }


def test_validate_scoreboard_single_invalid_bbox_uses_sentinel():
    result = validate_structured_prediction(
        _sample(),
        raw_output='{"text":"1-0","bbox":[10,20,30]}',
        structured_prediction={"text": "1-0", "bbox": [10, 20, 30]},
    )

    assert result.errors == []
    assert result.warnings == ["bbox invalid; using sentinel bbox"]
    assert result.structured_prediction == {
        "text": "1-0",
        "bbox": [-1.0, -1.0, -1.0, -1.0],
    }


def test_evaluate_scoreboard_single_sentinel_bbox_gets_zero_iou_and_fails_box():
    record = _record(
        {
            "text": "1-0",
            "bbox": [-1.0, -1.0, -1.0, -1.0],
        }
    )

    evaluation = evaluate_sample(
        _sample(),
        record,
        judge_client=StaticJudgeClient(always_pass=True),
    )

    assert evaluation.component_metrics["bbox_iou"] == 0.0
    assert evaluation.component_pass["bbox_pass"] == 0
    assert evaluation.component_pass["judge_pass"] == 1
    assert evaluation.task_pass == 0
