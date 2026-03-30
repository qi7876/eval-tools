import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnichain_eval.constants import JUDGE_REQUIRED_TASKS
from omnichain_eval.judge import (
    OpenAIJudgeClient,
    JudgeResponseFormatExhaustedError,
    JudgePromptTemplateError,
    load_judge_prompt_pack,
    render_judge_prompt,
)


JUDGE_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "judge_v1"


def _completion(*responses: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=response)) for response in responses
        ]
    )


def test_load_judge_prompt_pack_requires_all_judge_task_templates(tmp_path):
    prompt_root = tmp_path / "judge_prompts"
    prompt_root.mkdir()
    (prompt_root / "Scoreboard_Single.md").write_text(
        "Question: {{question_text}}\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgePromptTemplateError, match="missing prompt template"):
        load_judge_prompt_pack(prompt_root)


def test_load_judge_prompt_pack_rejects_unknown_variables(tmp_path):
    for task_name in sorted(JUDGE_REQUIRED_TASKS):
        (tmp_path / f"{task_name}.md").write_text(
            "Question: {{question_text}}\n",
            encoding="utf-8",
        )
    (tmp_path / "AI_Coach.md").write_text(
        "Bad: {{task_name}} {{question_text}}\n",
        encoding="utf-8",
    )

    with pytest.raises(JudgePromptTemplateError, match="unsupported variable"):
        load_judge_prompt_pack(tmp_path)


def test_render_judge_prompt_uses_task_specific_markdown():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Objects_Spatial_Relationships",
        question_text="What is the relation between Player A and Player B?",
        reference_payload={"text": "Player A is to the right of Player B."},
        prediction_payload={"text": "Player A is to the right of Player B."},
    )

    assert "spatial relationship between the queried objects" in prompt_text
    assert "camera-view definitions" in prompt_text
    assert "`in front of` / `behind`" in prompt_text
    assert "not who is leading in the play" in prompt_text
    assert "bounding boxes" not in prompt_text
    assert "task_specific_rule" not in prompt_text
    assert "task_name" not in prompt_text


def test_render_judge_prompt_for_ai_coach_focuses_on_mistake_identification():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="AI_Coach",
        question_text="What did the player do wrong in this possession?",
        reference_payload={"text": "The player dribbled too high and exposed the ball."},
        prediction_payload={"text": "He kept the dribble too loose and too high."},
    )

    assert "identify the player's mistakes" in prompt_text
    assert "ignore that advice unless it conflicts with or replaces the mistake identification" in prompt_text
    assert "additional mistake points" in prompt_text
    assert "coaching advice" not in prompt_text


def test_render_judge_prompt_for_scoreboard_single_ignores_bbox_and_does_not_emphasize_single():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Scoreboard_Single",
        question_text="What is the home team's score?",
        reference_payload={"text": "2"},
        prediction_payload={"text": "2"},
    )

    assert "information read from a scoreboard" in prompt_text
    assert "bounding box" not in prompt_text
    assert "single scoreboard fact" not in prompt_text


def test_render_judge_prompt_for_actions_focuses_on_actions_and_temporal_alignment():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Continuous_Actions_Caption",
        question_text="Describe the target athlete's actions over time.",
        reference_payload={"reference_segments": [{"start_frame": 0, "end_frame": 10, "text": "runs"}]},
        prediction_payload={"prediction_segments": [{"start_frame": 0, "end_frame": 10, "text": "runs"}]},
    )

    assert "action descriptions over time" in prompt_text
    assert "temporal alignment" in prompt_text
    assert "judging reference, not an exact segment template" in prompt_text
    assert "different split/merge of segments" in prompt_text
    assert "Do not require exact interval boundaries" in prompt_text
    assert "tracking" not in prompt_text.lower()


def test_render_judge_prompt_for_events_allows_equivalent_segmentation():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Continuous_Events_Caption",
        question_text="Describe the key events over time.",
        reference_payload={"reference_segments": [{"start_frame": 0, "end_frame": 10, "text": "a score happens"}]},
        prediction_payload={"prediction_segments": [{"start_frame": 0, "end_frame": 8, "text": "a score is made"}]},
    )

    assert "event descriptions over time" in prompt_text
    assert "judging reference, not an exact segment template" in prompt_text
    assert "different split/merge of segments" in prompt_text
    assert "rough temporal correspondence" in prompt_text


def test_render_judge_prompt_for_score_prediction_uses_general_game_answer_wording():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Score_Prediction",
        question_text="Who is more likely to finish ahead?",
        reference_payload={"text": "Team A"},
        prediction_payload={"text": "Team A"},
    )

    assert "game-related answer required by the question" in prompt_text
    assert "wrong ranking" in prompt_text
    assert "future result" not in prompt_text


def test_render_judge_prompt_for_temporal_causal_uses_result_reasoning_wording():
    prompt_text = render_judge_prompt(
        load_judge_prompt_pack(JUDGE_PROMPT_ROOT),
        task_name="Temporal_Causal",
        question_text="Why did Team A lose the match?",
        reference_payload={"text": "Team A conceded twice late in the game."},
        prediction_payload={"text": "They gave up two late goals."},
    )

    assert "why a result happened" in prompt_text
    assert "win, loss, ranking, lead change, failure, or another competition outcome" in prompt_text
    assert "restates the result without giving its cause" in prompt_text
    assert "side effect, consequence, or loosely related event" in prompt_text


def test_openai_judge_retries_invalid_json_before_accepting_valid_response(monkeypatch):
    responses = [
        _completion("not-json"),
        _completion(
            json.dumps(
                {
                    "correctness": 1,
                    "completeness": 1,
                    "faithfulness": 1,
                    "final_pass": 1,
                    "confidence": "high",
                    "brief_reason": "Looks good.",
                }
            )
        ),
    ]
    created: dict[str, object] = {}

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object):
            response = responses[self.calls]
            self.calls += 1
            return response

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())
            created["client"] = self

    monkeypatch.setattr("omnichain_eval.judge.OpenAI", FakeOpenAI)

    judge_client = OpenAIJudgeClient(
        base_url="http://judge.example/v1",
        api_key="dummy",
        prompt_root=JUDGE_PROMPT_ROOT,
        invalid_json_retries=1,
    )

    decision = judge_client.judge(
        task_name="Continuous_Events_Caption",
        question_text="Describe the events.",
        reference_payload={"reference_segments": [{"text": "A score."}]},
        prediction_payload={"prediction_segments": [{"text": "A score."}]},
    )

    assert decision.final_pass == 1
    assert decision.correctness == 1
    assert created["client"].chat.completions.calls == 2


def test_openai_judge_retries_schema_errors(monkeypatch):
    responses = [
        _completion(json.dumps({"final_pass": 1, "brief_reason": "missing keys"})),
        _completion(
            json.dumps(
                {
                    "correctness": 1,
                    "completeness": 1,
                    "faithfulness": 1,
                    "final_pass": 1,
                    "confidence": "high",
                    "brief_reason": "Looks good.",
                }
            )
        ),
    ]
    created: dict[str, object] = {}

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object):
            response = responses[self.calls]
            self.calls += 1
            return response

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())
            created["client"] = self

    monkeypatch.setattr("omnichain_eval.judge.OpenAI", FakeOpenAI)

    judge_client = OpenAIJudgeClient(
        base_url="http://judge.example/v1",
        api_key="dummy",
        prompt_root=JUDGE_PROMPT_ROOT,
        invalid_json_retries=1,
    )

    decision = judge_client.judge(
        task_name="Continuous_Events_Caption",
        question_text="Describe the events.",
        reference_payload={"reference_segments": [{"text": "A score."}]},
        prediction_payload={"prediction_segments": [{"text": "A score."}]},
    )

    assert decision.final_pass == 1
    assert created["client"].chat.completions.calls == 2


def test_openai_judge_raises_after_exhausting_format_retries(monkeypatch):
    responses = [
        _completion(json.dumps({"wrong_key": 1})),
        _completion(json.dumps({"still_wrong": 2})),
    ]
    created: dict[str, object] = {}

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_: object):
            response = responses[self.calls]
            self.calls += 1
            return response

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())
            created["client"] = self

    monkeypatch.setattr("omnichain_eval.judge.OpenAI", FakeOpenAI)

    judge_client = OpenAIJudgeClient(
        base_url="http://judge.example/v1",
        api_key="dummy",
        prompt_root=JUDGE_PROMPT_ROOT,
        invalid_json_retries=1,
    )

    with pytest.raises(JudgeResponseFormatExhaustedError, match="did not match schema"):
        judge_client.judge(
            task_name="Continuous_Events_Caption",
            question_text="Describe the events.",
            reference_payload={"reference_segments": [{"text": "A score."}]},
            prediction_payload={"prediction_segments": [{"text": "A score."}]},
        )

    assert created["client"].chat.completions.calls == 2


def test_openai_judge_forwards_extra_body(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object):
            captured.update(kwargs)
            return _completion(
                json.dumps(
                    {
                        "correctness": 1,
                        "completeness": 1,
                        "faithfulness": 1,
                        "final_pass": 1,
                        "confidence": "high",
                        "brief_reason": "Looks good.",
                    }
                )
            )

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("omnichain_eval.judge.OpenAI", FakeOpenAI)

    judge_client = OpenAIJudgeClient(
        base_url="http://judge.example/v1",
        api_key="dummy",
        prompt_root=JUDGE_PROMPT_ROOT,
        extra_body={"thinking": {"type": "disabled"}},
    )

    decision = judge_client.judge(
        task_name="Continuous_Events_Caption",
        question_text="Describe the events.",
        reference_payload={"reference_segments": [{"text": "A score."}]},
        prediction_payload={"prediction_segments": [{"text": "A score."}]},
    )

    assert decision.final_pass == 1
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
