from pathlib import Path

import pytest

from omnichain_eval.constants import ALL_TASKS
from omnichain_eval.prepare import load_prepared_samples
from omnichain_eval.prompting import (
    PromptTemplateError,
    build_chain_history,
    build_model_input,
    load_prompt_pack,
    render_prompt,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"


def test_load_prompt_pack_requires_all_task_templates(tmp_path):
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    (prompt_root / "Scoreboard_Single.md").write_text(
        "Question: {{question}}\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="missing prompt template"):
        load_prompt_pack(prompt_root)


def test_load_prompt_pack_rejects_unknown_variables(tmp_path):
    for task_name in sorted(ALL_TASKS):
        (tmp_path / f"{task_name}.md").write_text(
            "Question: {{question}}\n",
            encoding="utf-8",
        )
    (tmp_path / "AI_Coach.md").write_text(
        "Bad: {{unknown_variable}}\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="unsupported variable"):
        load_prompt_pack(tmp_path)


def test_rendered_spatial_imagination_model_input_includes_history(monkeypatch, tmp_path):
    from PIL import Image
    from omnichain_eval.prepare import build_prepared_data

    def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
        return {
            frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
            for frame_index in frame_indices
        }

    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    prepared_samples = load_prepared_samples(prepared_root, "main")
    prompt_pack = load_prompt_pack(PROMPT_ROOT)

    upstream_sample = next(sample for sample in prepared_samples if sample.sample_id == "TestSport/TestEvent/1#2")
    downstream_sample = next(sample for sample in prepared_samples if sample.sample_id == "TestSport/TestEvent/1#4")
    conversation_history = build_chain_history(
        upstream_sample,
        {"text": "mock upstream answer"},
    )
    model_input = build_model_input(
        downstream_sample,
        render_prompt(prompt_pack, downstream_sample),
        conversation_history=conversation_history,
    )

    assert [message.role for message in model_input.messages] == ["user", "assistant", "user"]
    assert model_input.messages[0].content == upstream_sample.question_text
    assert model_input.messages[1].content == '{"text": "mock upstream answer"}'
