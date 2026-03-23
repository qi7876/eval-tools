from pathlib import Path

import pytest

from omnichain_eval.schema import PreparedSample
from omnichain_eval.structurer import (
    StructurerBackend,
    StructurerResponseFormatExhaustedError,
    StructurerService,
)
from omnichain_eval.template_pack import TaskTemplate


def _sample() -> PreparedSample:
    return PreparedSample(
        sample_id="video/sample#1",
        annotation_id="1",
        video_key="video/sample",
        task_name="Scoreboard_Multiple",
        task_level="independent",
        protocol_id="main",
        question_text="Who is leading?",
        sampled_frames_original=[0, 10, 20],
        sampled_to_original={0: 0, 1: 10, 2: 20},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "Team A is leading."},
    )


def _prompt_pack() -> dict[str, TaskTemplate]:
    return {
        "Scoreboard_Multiple": TaskTemplate(
            task_name="Scoreboard_Multiple",
            path=Path("Scoreboard_Multiple.md"),
            system_template="",
            user_template="{{question}}\n{{raw_output}}\n{{output_schema}}",
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
