import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnichain_eval.judge import OpenAIJudgeClient, JudgeResponseFormatExhaustedError


JUDGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_v1.md"


def _completion(*responses: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=response)) for response in responses
        ]
    )


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
        prompt_path=JUDGE_PROMPT_PATH,
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
        prompt_path=JUDGE_PROMPT_PATH,
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
        prompt_path=JUDGE_PROMPT_PATH,
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
