from omnichain_eval.judge import StaticJudgeClient
from omnichain_eval.metrics import evaluate_sample
from omnichain_eval.normalize import validate_structured_prediction
from omnichain_eval.schema import PreparedSample, StructuredPredictionRecord


def _sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/sample#1",
        annotation_id="1",
        video_key="video/sample",
        task_name="Objects_Spatial_Relationships",
        task_level="independent",
        protocol_id="main",
        question_text="What is the spatial relationship between Player A and Player B?",
        sampled_frames_original=[12],
        sampled_to_original={0: 12},
        frame_files=["frames/0000.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player A", "bbox": [10.0, 20.0, 30.0, 40.0]},
                {"label": "Player B", "bbox": [50.0, 60.0, 70.0, 80.0]},
            ],
        },
    )


def _record(structured_prediction: dict | None, *, raw_output: str = "{}") -> StructuredPredictionRecord:
    return StructuredPredictionRecord(
        sample_id="video/sample#1",
        task_name="Objects_Spatial_Relationships",
        video_key="video/sample",
        protocol_id="main",
        raw_output=raw_output,
        structured_prediction=structured_prediction,
        structuring_errors=[],
        structuring_warnings=[],
    )


def test_validate_object_spatial_accepts_reversed_output_order():
    sample = _sample()

    result = validate_structured_prediction(
        sample,
        raw_output='{"text":"right of"}',
        structured_prediction={
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
            ],
        },
    )

    assert result.errors == []
    assert result.structured_prediction == {
        "text": "Player A is to the right of Player B.",
        "objects": [
            {"label": "Player A", "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"label": "Player B", "bbox": [50.0, 60.0, 70.0, 80.0]},
        ],
    }


def test_validate_object_spatial_rejects_duplicate_missing_and_extra_labels():
    sample = _sample()

    duplicate = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "right of",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
                {"label": "Player A", "bbox": [50, 60, 70, 80]},
            ],
        },
    )
    assert "duplicate object label: Player A" in duplicate.errors

    extra = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "right of",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
                {"label": "Player C", "bbox": [1, 2, 3, 4]},
            ],
        },
    )
    assert "unexpected object label: Player C" in extra.errors


def test_validate_object_spatial_uses_sentinel_for_missing_label():
    sample = _sample()

    result = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "right of",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
            ],
        },
    )

    assert result.errors == []
    assert result.structured_prediction == {
        "text": "right of",
        "objects": [
            {"label": "Player A", "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"label": "Player B", "bbox": [-1.0, -1.0, -1.0, -1.0]},
        ],
    }
    assert "missing object label: Player B; using sentinel bbox" in result.warnings


def test_validate_object_spatial_uses_sentinel_for_invalid_or_missing_bbox():
    sample = _sample()

    invalid = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "right of",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 1300, 40]},
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
            ],
        },
    )

    assert invalid.errors == []
    assert invalid.structured_prediction == {
        "text": "right of",
        "objects": [
            {"label": "Player A", "bbox": [-1.0, -1.0, -1.0, -1.0]},
            {"label": "Player B", "bbox": [50.0, 60.0, 70.0, 80.0]},
        ],
    }
    assert "objects[0].bbox invalid; using sentinel bbox" in invalid.warnings

    missing = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "right of",
            "objects": [
                {"label": "Player A"},
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
            ],
        },
    )

    assert missing.errors == []
    assert missing.structured_prediction == {
        "text": "right of",
        "objects": [
            {"label": "Player A", "bbox": [-1.0, -1.0, -1.0, -1.0]},
            {"label": "Player B", "bbox": [50.0, 60.0, 70.0, 80.0]},
        ],
    }
    assert "objects[0].bbox missing; using sentinel bbox" in missing.warnings


def test_evaluate_object_spatial_matches_boxes_by_label():
    sample = _sample()
    record = _record(
        {
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
            ],
        }
    )

    evaluation = evaluate_sample(
        sample,
        record,
        judge_client=StaticJudgeClient(always_pass=True),
    )

    assert evaluation.component_metrics["object_ious"] == {
        "Player A": 1.0,
        "Player B": 1.0,
    }
    assert evaluation.component_pass["object_passes"] == {
        "Player A": 1,
        "Player B": 1,
    }
    assert evaluation.component_pass["judge_pass"] == 1
    assert evaluation.task_pass == 1


def test_evaluate_object_spatial_fails_when_required_label_is_missing():
    sample = _sample()
    validation = validate_structured_prediction(
        sample,
        raw_output="{}",
        structured_prediction={
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
            ],
        },
    )

    assert validation.errors == []
    assert validation.structured_prediction is not None
    assert validation.structured_prediction["objects"][1]["bbox"] == [-1.0, -1.0, -1.0, -1.0]
    assert "missing object label: Player B; using sentinel bbox" in validation.warnings


def test_evaluate_object_spatial_keeps_text_judgment_but_fails_bbox_component_for_sentinel():
    sample = _sample()
    record = _record(
        {
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
                {"label": "Player B", "bbox": [-1, -1, -1, -1]},
            ],
        }
    )

    evaluation = evaluate_sample(
        sample,
        record,
        judge_client=StaticJudgeClient(always_pass=True),
    )

    assert evaluation.component_metrics["object_ious"] == {
        "Player A": 1.0,
        "Player B": 0.0,
    }
    assert evaluation.component_pass["object_passes"] == {
        "Player A": 1,
        "Player B": 0,
    }
    assert evaluation.component_pass["judge_pass"] == 1
    assert evaluation.task_pass == 0
