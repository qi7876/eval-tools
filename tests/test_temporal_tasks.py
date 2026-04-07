from omnichain_eval.judge import JudgeClient
from omnichain_eval.metrics import evaluate_sample
from omnichain_eval.schema import JudgeDecision, PreparedSample, StructuredPredictionRecord


class CapturingJudge(JudgeClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def judge(
        self,
        task_name: str,
        question_text: str,
        reference_payload: dict,
        prediction_payload: dict,
    ) -> JudgeDecision:
        self.calls.append(
            {
                "task_name": task_name,
                "question_text": question_text,
                "reference_payload": reference_payload,
                "prediction_payload": prediction_payload,
            }
        )
        return JudgeDecision(
            correctness=1,
            completeness=1,
            faithfulness=1,
            final_pass=1,
            confidence="high",
            brief_reason="Captured for test.",
            raw_response=None,
        )


def _record(
    sample: PreparedSample,
    structured_prediction: dict | None,
    *,
    raw_output: str = "{}",
) -> StructuredPredictionRecord:
    return StructuredPredictionRecord(
        sample_id=sample.sample_id,
        task_name=sample.task_name,
        video_key=sample.video_key,
        protocol_id=sample.protocol_id,
        raw_output=raw_output,
        structured_prediction=structured_prediction,
        structuring_errors=[],
        structuring_warnings=[],
    )


def _events_sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/events#1",
        annotation_id="1",
        video_key="video/events",
        task_name="Continuous_Events_Caption",
        task_level="independent",
        protocol_id="main",
        question_text="Describe the key events over time.",
        sampled_frames_original=[100, 110, 120, 130, 140],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130, 4: 140},
        frame_files=[f"frames/{index:04d}.jpg" for index in range(5)],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "segments_original": [
                {"start_frame": 100, "end_frame": 130, "text": "original frames text"}
            ],
            "segments_sampled": [
                {"start_sampled": 1, "end_sampled": 3, "text": "sampled frames text"}
            ],
        },
    )


def _actions_sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/actions#1",
        annotation_id="2",
        video_key="video/actions",
        task_name="Continuous_Actions_Caption",
        task_level="independent",
        protocol_id="main",
        question_text="Describe the athlete actions over time.",
        sampled_frames_original=[100, 110, 120, 130, 140],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130, 4: 140},
        frame_files=[f"frames/{index:04d}.jpg" for index in range(5)],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "segments_original": [
                {"start_frame": 100, "end_frame": 130, "text": "original action text"}
            ],
            "segments_sampled": [
                {"start_sampled": 1, "end_sampled": 3, "text": "sampled action text"}
            ],
            "tracking_gt_sampled": [
                {"frame_sampled": 1, "frame_original": 110, "bbox_mot": [10.0, 20.0, 30.0, 40.0]}
            ],
        },
    )


def _stg_sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/stg#1",
        annotation_id="3",
        video_key="video/stg",
        task_name="Spatial_Temporal_Grounding",
        task_level="independent",
        protocol_id="main",
        question_text="Find when the described action happens.",
        sampled_frames_original=[100, 110, 120, 130, 140],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130, 4: 140},
        frame_files=[f"frames/{index:04d}.jpg" for index in range(5)],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "time_window_original": [120, 140],
            "time_window_sampled": [2, 4],
            "tracking_gt_sampled": [
                {"frame_sampled": 2, "frame_original": 120, "bbox_mot": [10.0, 20.0, 30.0, 40.0]}
            ],
        },
    )


def test_evaluate_events_uses_sampled_segments_for_judge_and_bertscore():
    sample = _events_sample()
    judge = CapturingJudge()

    evaluation = evaluate_sample(
        sample,
        _record(
            sample,
            {
                "segments": [
                    {
                        "start_sampled": 48,
                        "end_sampled": 73,
                        "text": "predicted sampled event",
                    }
                ]
            },
        ),
        judge_client=judge,
    )

    assert judge.calls == [
        {
            "task_name": "Continuous_Events_Caption",
            "question_text": sample.question_text,
            "reference_payload": {"reference_segments": sample.reference_payload["segments_sampled"]},
            "prediction_payload": {
                "prediction_segments": [
                    {
                        "start_sampled": 48,
                        "end_sampled": 73,
                        "text": "predicted sampled event",
                    }
                ]
            },
        }
    ]
    assert evaluation.bertscore_reference == "sampled frames text"
    assert evaluation.bertscore_candidate == "predicted sampled event"
    assert evaluation.component_pass["judge_pass"] == 1
    assert evaluation.task_pass == 1


def test_evaluate_actions_keeps_out_of_range_segments_and_ignores_extra_tracking_frames():
    sample = _actions_sample()
    judge = CapturingJudge()

    evaluation = evaluate_sample(
        sample,
        _record(
            sample,
            {
                "segments": [
                    {
                        "start_sampled": 48,
                        "end_sampled": 73,
                        "text": "predicted sampled action",
                    }
                ],
                "tracking": [
                    {"frame_sampled": 1, "bbox_mot": [10.0, 20.0, 30.0, 40.0]},
                    {"frame_sampled": 99, "bbox_mot": [10.0, 20.0, 30.0, 40.0]},
                ],
            },
        ),
        judge_client=judge,
    )

    assert judge.calls[0]["reference_payload"] == {
        "reference_segments": sample.reference_payload["segments_sampled"]
    }
    assert judge.calls[0]["prediction_payload"] == {
        "prediction_segments": [
            {
                "start_sampled": 48,
                "end_sampled": 73,
                "text": "predicted sampled action",
            }
        ]
    }
    assert evaluation.component_metrics["tracking_mean_iou"] == 1.0
    assert evaluation.component_metrics["tracking_pass_rate"] == 1.0
    assert evaluation.component_pass["tracking_pass"] == 1
    assert evaluation.task_pass == 1


def test_evaluate_stg_uses_unclipped_sampled_window_tiou():
    sample = _stg_sample()

    evaluation = evaluate_sample(
        sample,
        _record(
            sample,
            {
                "time_window_sampled": [4, 8],
                "tracking": [{"frame_sampled": 2, "bbox_mot": [10.0, 20.0, 30.0, 40.0]}],
            },
        ),
        judge_client=None,
        oracle_upstream=True,
    )

    assert evaluation.structuring_errors == []
    assert evaluation.component_metrics["tiou"] == 1.0 / 7.0
    assert evaluation.component_pass["tiou_pass"] == 0
    assert evaluation.task_pass == 0


def test_evaluate_stg_reversed_window_scores_zero_in_evaluation():
    sample = _stg_sample()

    evaluation = evaluate_sample(
        sample,
        _record(
            sample,
            {
                "time_window_sampled": [8, 4],
                "tracking": [{"frame_sampled": 2, "bbox_mot": [10.0, 20.0, 30.0, 40.0]}],
            },
        ),
        judge_client=None,
        oracle_upstream=True,
    )

    assert evaluation.component_metrics["tiou"] == 0.0
    assert evaluation.component_pass["tiou_pass"] == 0
    assert "time_window_sampled start > end: 8-4" in evaluation.structuring_errors
