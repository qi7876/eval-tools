from pathlib import Path
from types import SimpleNamespace

import pytest

from omnichain_eval.schema import PreparedSample
from omnichain_eval.structurer import (
    OpenAIStructurerBackend,
    StructurerBackend,
    StructurerResponseFormatExhaustedError,
    StructurerService,
    load_oracle_structurer_prompt_pack,
    load_structurer_prompt_pack,
    render_structurer_prompt,
)
from omnichain_eval.template_pack import TaskTemplate

PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "structurer_v1"
ORACLE_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "structurer_oracle_v1"
ALL_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def _sample(
    *,
    task_name: str = "Scoreboard_Multiple",
    question_text: str = "Who is leading?",
    reference_payload: dict | None = None,
) -> PreparedSample:
    return PreparedSample(
        sample_id="video/sample#1",
        annotation_id="1",
        video_key="video/sample",
        task_name=task_name,
        task_level="independent",
        protocol_id="main",
        question_text=question_text,
        sampled_frames_original=[0, 10, 20],
        sampled_to_original={0: 0, 1: 10, 2: 20},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload=reference_payload or {"text": "Team A is leading."},
    )


def _prompt_pack() -> dict[str, TaskTemplate]:
    return {
        "Scoreboard_Multiple": TaskTemplate(
            task_name="Scoreboard_Multiple",
            path=Path("Scoreboard_Multiple.md"),
            prompt_template='{{raw_output}}\n{"text": ""}',
        )
    }


class SequenceBackend(StructurerBackend):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def complete(self, *, sample, raw_output, rendered_prompt):
        response = self.responses[self.calls]
        self.calls += 1
        return [response]


class StaticParseBackend(StructurerBackend):
    def complete(self, *, sample, raw_output, rendered_prompt):
        return [raw_output]


def test_structurer_retries_invalid_json_before_accepting_valid_response():
    backend = SequenceBackend(
        [
            "not-json",
            '{"text": "Team A is leading."}',
        ]
    )
    service = StructurerService(
        backend=backend,
        prompt_pack=_prompt_pack(),
        invalid_json_retries=1,
    )

    result = service.structure(_sample(), '{"text": "Team A is leading."}')

    assert result.structured_prediction == {"text": "Team A is leading."}
    assert backend.calls == 2


def test_structurer_retries_schema_errors_before_accepting_valid_response():
    backend = SequenceBackend(
        [
            '{"wrong_key": "value"}',
            '{"text": "Team A is leading."}',
        ]
    )
    service = StructurerService(
        backend=backend,
        prompt_pack=_prompt_pack(),
        invalid_json_retries=1,
    )

    result = service.structure(_sample(), '{"text": "Team A is leading."}')

    assert result.structured_prediction == {"text": "Team A is leading."}
    assert backend.calls == 2


def test_structurer_raises_after_exhausting_format_retries():
    backend = SequenceBackend(
        [
            '{"wrong_key": "value"}',
            '{"still_wrong": "value"}',
        ]
    )
    service = StructurerService(
        backend=backend,
        prompt_pack=_prompt_pack(),
        invalid_json_retries=1,
    )

    with pytest.raises(StructurerResponseFormatExhaustedError, match="missing text field"):
        service.structure(_sample(), '{"text": "Team A is leading."}')

    assert backend.calls == 2


def test_structurer_logs_prompt_and_response_on_failure(capsys):
    backend = SequenceBackend(
        [
            '{"wrong_key": "value"}',
            '{"still_wrong": "value"}',
        ]
    )
    service = StructurerService(
        backend=backend,
        prompt_pack=_prompt_pack(),
        invalid_json_retries=1,
    )

    with pytest.raises(StructurerResponseFormatExhaustedError, match="missing text field"):
        service.structure(_sample(), '{"text": "Team A is leading."}')

    captured = capsys.readouterr()
    assert "[Structurer Debug] failure detected" in captured.err
    assert "sample_id=video/sample#1" in captured.err
    assert "missing text field after 2 attempt(s)" in captured.err
    assert '{"text": "Team A is leading."}' in captured.err
    assert '{"still_wrong": "value"}' in captured.err


def test_rendered_structurer_prompt_for_scoreboard_single_includes_bbox_rules():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Scoreboard_Single",
            question_text="Read the scoreboard and localize it.",
        ),
        'Reasoning...\nFinal answer: {"text": "1-0", "bbox": [10, 20, 30, 40]}',
    )

    assert "Convert the raw model output into the canonical JSON schema." in rendered.prompt_text
    assert "extract the final answer" in rendered.prompt_text
    assert "bbox = [x1, y1, x2, y2]" in rendered.prompt_text
    assert "bbox = [-1, -1, -1, -1]" in rendered.prompt_text
    assert "normalized_1000 coordinate system" in rendered.prompt_text
    assert "explicit valid scoreboard bbox" in rendered.prompt_text
    assert "Question:" not in rendered.prompt_text


def test_rendered_structurer_prompt_for_actions_includes_tracking_rules():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Continuous_Actions_Caption",
            question_text="Describe the athlete actions over time.",
        ),
        "frames 2-4: starts running; frame 3 bbox [1,2,3,4]",
    )

    assert "normalize explicit interval expressions" in rendered.prompt_text
    assert "normalize explicit tracking rows or coordinate strings" in rendered.prompt_text
    assert "`bbox_mot` must be formatted as normalized_1000 `[left, top, width, height]`." in rendered.prompt_text
    assert "(1000, 1000)" in rendered.prompt_text
    assert "Question:" not in rendered.prompt_text


def test_rendered_structurer_prompt_for_events_includes_full_segment_schema():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Continuous_Events_Caption",
            question_text="Describe the key events over time.",
        ),
        "frames 0-1: a player shoots",
    )

    assert '\"start_sampled\": 0' in rendered.prompt_text
    assert '\"end_sampled\": 3' in rendered.prompt_text
    assert '\"text\": \"\"' in rendered.prompt_text
    assert "Each segment must contain `start_sampled`, `end_sampled`, and `text`." in rendered.prompt_text


def test_rendered_structurer_prompt_for_stg_includes_full_tracking_schema():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Spatial_Temporal_Grounding",
            question_text="Find when the described action happens.",
        ),
        "window [0,2], frame 1 bbox [1,2,3,4]",
    )

    assert '\"time_window_sampled\": [0, 4]' in rendered.prompt_text
    assert '\"frame_sampled\": 0' in rendered.prompt_text
    assert '\"bbox_mot\": [0, 0, 100, 100]' in rendered.prompt_text
    assert "`time_window_sampled` must be either an empty list or a two-value list `[start_sampled, end_sampled]`." in rendered.prompt_text
    assert "Each tracking row must contain `frame_sampled` and `bbox_mot`." in rendered.prompt_text


def test_rendered_oracle_structurer_prompt_for_actions_includes_full_segment_schema():
    prompt_pack = load_oracle_structurer_prompt_pack(ORACLE_PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Continuous_Actions_Caption",
            question_text="Describe the athlete actions over time.",
        ),
        "frames 0-1: runs",
        oracle_upstream=True,
    )

    assert '\"start_sampled\": 0' in rendered.prompt_text
    assert '\"end_sampled\": 3' in rendered.prompt_text
    assert '\"text\": \"\"' in rendered.prompt_text
    assert "Each segment must contain `start_sampled`, `end_sampled`, and `text`." in rendered.prompt_text
    assert '\"tracking\"' not in rendered.prompt_text


def test_rendered_oracle_structurer_prompt_for_stg_includes_explicit_window_schema():
    prompt_pack = load_oracle_structurer_prompt_pack(ORACLE_PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Spatial_Temporal_Grounding",
            question_text="Ground the action.",
        ),
        "frames 0-2",
        oracle_upstream=True,
    )

    assert '\"time_window_sampled\": [0, 4]' in rendered.prompt_text
    assert "`time_window_sampled` must be either an empty list or a two-value list `[start_sampled, end_sampled]`." in rendered.prompt_text
    assert '\"tracking\"' not in rendered.prompt_text


def test_rendered_structurer_prompt_for_text_task_is_pure_extractor():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Temporal_Causal",
            question_text="Why did the play fail?",
        ),
        "Analysis...\nFinal answer: the defender blocked the lane.",
    )

    assert "Use only information that explicitly appears in the raw model output." in rendered.prompt_text
    assert "prefer the last one presented as the final answer." in rendered.prompt_text
    assert "Question:" not in rendered.prompt_text
    assert "{{question}}" not in rendered.prompt_text


def test_rendered_structurer_prompt_for_ai_coach_uses_mistake_labels_not_suggestion():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="AI_Coach",
            question_text="What did the player do wrong?",
        ),
        "Final answer: the player exposed the ball and lost balance.",
    )

    assert '"mistake"' in rendered.prompt_text
    assert '"error"' in rendered.prompt_text
    assert '"suggestion"' not in rendered.prompt_text


def test_rendered_structurer_prompt_for_score_prediction_uses_general_answer_language():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Score_Prediction",
            question_text="Who is more likely to finish ahead?",
        ),
        "Final answer: Team A is more likely to finish ahead.",
    )

    assert '"answer"' in rendered.prompt_text
    assert '"conclusion"' in rendered.prompt_text
    assert "candidate answers" in rendered.prompt_text
    assert "candidate predictions" not in rendered.prompt_text
    assert '"winner"' not in rendered.prompt_text


def test_rendered_structurer_prompt_for_objects_spatial_includes_label_rules():
    prompt_pack = load_structurer_prompt_pack(PROMPT_ROOT)
    rendered = render_structurer_prompt(
        prompt_pack,
        _sample(
            task_name="Objects_Spatial_Relationships",
            question_text="What is the spatial relationship between Player A and Player B?",
            reference_payload={
                "text": "Player A is to the right of Player B.",
                "objects": [
                    {"label": "Player A", "bbox": [10, 20, 30, 40]},
                    {"label": "Player B", "bbox": [50, 60, 70, 80]},
                ],
            },
        ),
        'Final answer: {"text": "right of", "objects": [{"label": "Player B", "bbox": [1,2,3,4]}]}',
    )

    assert "Use exactly these object labels" in rendered.prompt_text
    assert '["Player A", "Player B"]' in rendered.prompt_text
    assert "Output exactly one object entry for each required label." in rendered.prompt_text
    assert "Match boxes by explicit label, not by first/second position." in rendered.prompt_text
    assert "bbox = [-1, -1, -1, -1]" in rendered.prompt_text
    assert "Question:" not in rendered.prompt_text


def test_normal_structurer_for_object_spatial_requires_bbox_field_without_prompt_rewrite():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    with pytest.raises(
        StructurerResponseFormatExhaustedError,
        match="objects\\[0\\] must contain label and bbox after 1 attempt\\(s\\)",
    ):
        service.structure(
            _sample(
                task_name="Objects_Spatial_Relationships",
                question_text="What is the spatial relationship between Player A and Player B?",
                reference_payload={
                    "text": "Player A is to the right of Player B.",
                    "objects": [
                        {"label": "Player A", "bbox": [10, 20, 30, 40]},
                        {"label": "Player B", "bbox": [50, 60, 70, 80]},
                    ],
                },
            ),
            '{"text": "right of", "objects": [{"label": "Player A"}]}',
        )


def test_load_structurer_prompt_pack_rejects_removed_context_variables(tmp_path):
    for task_name in [
        "AI_Coach",
        "Continuous_Actions_Caption",
        "Continuous_Events_Caption",
        "Objects_Spatial_Relationships",
        "Score_Prediction",
        "Scoreboard_Multiple",
        "Scoreboard_Single",
        "Spatial_Imagination",
        "Spatial_Temporal_Grounding",
        "Temporal_Causal",
    ]:
        (tmp_path / f"{task_name}.md").write_text(
            "Bad: {{task_name}} {{question}} {{num_sampled_frames}} {{sampled_index_range}}\n",
            encoding="utf-8",
        )

    with pytest.raises(Exception, match="unsupported variable"):
        load_structurer_prompt_pack(tmp_path)


def test_normal_structurer_for_actions_requires_tracking():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    with pytest.raises(
        StructurerResponseFormatExhaustedError,
        match="tracking must be a list after 1 attempt\\(s\\)",
    ):
        service.structure(
            _sample(
                task_name="Continuous_Actions_Caption",
                question_text="Describe the athlete actions over time.",
            ),
            '{"segments": [{"start_sampled": 0, "end_sampled": 1, "text": "runs"}]}',
        )


def test_normal_structurer_for_stg_requires_tracking():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    with pytest.raises(
        StructurerResponseFormatExhaustedError,
        match="tracking must be a list after 1 attempt\\(s\\)",
    ):
        service.structure(
            _sample(
                task_name="Spatial_Temporal_Grounding",
                question_text="Ground the action.",
            ),
            '{"time_window_sampled": [0, 1]}',
        )


def test_normal_structurer_for_events_accepts_out_of_range_segments():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    result = service.structure(
        _sample(
            task_name="Continuous_Events_Caption",
            question_text="Describe the events over time.",
        ),
        '{"segments": [{"start_sampled": 48, "end_sampled": 73, "text": "player shoots"}]}',
    )

    assert result.errors == []
    assert result.structured_prediction == {
        "segments": [{"start_sampled": 48, "end_sampled": 73, "text": "player shoots"}]
    }


def test_normal_structurer_for_actions_accepts_out_of_range_segment_and_tracking_indices():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    result = service.structure(
        _sample(
            task_name="Continuous_Actions_Caption",
            question_text="Describe the athlete actions over time.",
        ),
        '{"segments": [{"start_sampled": 48, "end_sampled": 73, "text": "player runs"}], "tracking": [{"frame_sampled": 73, "bbox_mot": [1, 2, 3, 4]}]}',
    )

    assert result.errors == []
    assert result.structured_prediction == {
        "segments": [{"start_sampled": 48, "end_sampled": 73, "text": "player runs"}],
        "tracking": [{"frame_sampled": 73, "bbox_mot": [1.0, 2.0, 3.0, 4.0]}],
    }


def test_normal_structurer_for_stg_accepts_out_of_range_window_and_tracking_indices():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    result = service.structure(
        _sample(
            task_name="Spatial_Temporal_Grounding",
            question_text="Ground the action.",
        ),
        '{"time_window_sampled": [48, 73], "tracking": [{"frame_sampled": 73, "bbox_mot": [1, 2, 3, 4]}]}',
    )

    assert result.errors == []
    assert result.structured_prediction == {
        "time_window_sampled": [48, 73],
        "tracking": [{"frame_sampled": 73, "bbox_mot": [1.0, 2.0, 3.0, 4.0]}],
    }


def test_oracle_structurer_for_actions_accepts_segments_only():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        oracle_prompt_pack=load_oracle_structurer_prompt_pack(ORACLE_PROMPT_ROOT),
        invalid_json_retries=0,
    )

    result = service.structure(
        _sample(
            task_name="Continuous_Actions_Caption",
            question_text="Describe the athlete actions over time.",
        ),
        '{"segments": [{"start_sampled": 0, "end_sampled": 1, "text": "runs"}]}',
        oracle_upstream=True,
    )

    assert result.structured_prediction == {
        "segments": [{"start_sampled": 0, "end_sampled": 1, "text": "runs"}]
    }


def test_oracle_structurer_for_stg_accepts_time_window_only():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        oracle_prompt_pack=load_oracle_structurer_prompt_pack(ORACLE_PROMPT_ROOT),
        invalid_json_retries=0,
    )

    result = service.structure(
        _sample(
            task_name="Spatial_Temporal_Grounding",
            question_text="Ground the action.",
        ),
        '{"time_window_sampled": [0, 1]}',
        oracle_upstream=True,
    )

    assert result.structured_prediction == {"time_window_sampled": [0, 1]}


def test_structurer_rejects_legacy_object_spatial_bbox_a_bbox_b_schema():
    service = StructurerService(
        backend=StaticParseBackend(),
        prompt_pack=load_structurer_prompt_pack(PROMPT_ROOT),
        invalid_json_retries=0,
    )

    with pytest.raises(
        StructurerResponseFormatExhaustedError,
        match="objects must be a list after 1 attempt\\(s\\)",
    ):
        service.structure(
            _sample(
                task_name="Objects_Spatial_Relationships",
                question_text="What is the spatial relationship between Player A and Player B?",
                reference_payload={
                    "text": "Player A is to the right of Player B.",
                    "objects": [
                        {"label": "Player A", "bbox": [10, 20, 30, 40]},
                        {"label": "Player B", "bbox": [50, 60, 70, 80]},
                    ],
                },
            ),
            '{"text": "right of", "bbox_a": [10, 20, 30, 40], "bbox_b": [50, 60, 70, 80]}',
        )


def test_openai_structurer_forwards_extra_body(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"text": "Team A is leading."}')
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("omnichain_eval.structurer.OpenAI", FakeOpenAI)

    backend = OpenAIStructurerBackend(
        base_url="http://structurer.example/v1",
        api_key="dummy",
        temperature=0,
        extra_body={"provider_hint": "structurer"},
    )
    responses = backend.complete(
        sample=_sample(),
        raw_output='{"text": "Team A is leading."}',
        rendered_prompt=render_structurer_prompt(
            _prompt_pack(),
            _sample(),
            '{"text": "Team A is leading."}',
        ),
    )

    assert responses == ['{"text": "Team A is leading."}']
    assert captured["temperature"] == 0.0
    assert captured["extra_body"] == {
        "enable_thinking": False,
        "provider_hint": "structurer",
    }


def test_prompt_sources_do_not_use_incomplete_empty_array_schemas():
    bad_snippets = (
        '"segments": []',
        '"tracking": []',
        '"time_window_sampled": []',
        '"bbox": []',
        '"objects": []',
    )

    for path in ALL_PROMPTS_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for snippet in bad_snippets:
            assert snippet not in text, f"{path} still contains incomplete schema snippet: {snippet}"
