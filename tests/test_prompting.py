from pathlib import Path

import pytest

from omnichain_eval.constants import ALL_TASKS
from omnichain_eval.prepare import load_prepared_samples
from omnichain_eval.prompting import (
    PromptTemplateError,
    build_chain_history,
    build_model_input,
    load_oracle_prompt_pack,
    load_prompt_pack,
    render_oracle_upstream_prompt,
    render_prompt,
)
from omnichain_eval.schema import PreparedSample


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"
ORACLE_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_oracle_v1"


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


def test_load_prompt_pack_rejects_removed_metadata_variables(tmp_path):
    for task_name in sorted(ALL_TASKS):
        (tmp_path / f"{task_name}.md").write_text(
            "Bad: {{task_name}} {{protocol_id}} {{task_level}}\n",
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
    rendered_upstream_prompt = render_prompt(prompt_pack, upstream_sample)
    conversation_history = build_chain_history(
        rendered_upstream_prompt.prompt_text,
        {"text": "mock upstream answer"},
    )
    model_input = build_model_input(
        downstream_sample,
        render_prompt(prompt_pack, downstream_sample),
        conversation_history=conversation_history,
    )

    assert [message.role for message in model_input.messages] == ["user", "assistant", "user"]
    assert model_input.messages[0].content == rendered_upstream_prompt.prompt_text
    assert model_input.messages[1].content == '{"text": "mock upstream answer"}'


def test_render_prompt_omits_metadata_and_nonessential_indexing(monkeypatch, tmp_path):
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

    scoreboard_sample = next(
        sample for sample in prepared_samples if sample.task_name == "Scoreboard_Single"
    )
    actions_sample = next(
        sample for sample in prepared_samples if sample.task_name == "Continuous_Actions_Caption"
    )

    scoreboard_prompt = render_prompt(prompt_pack, scoreboard_sample).prompt_text
    actions_prompt = render_prompt(prompt_pack, actions_sample).prompt_text

    assert "Task:" not in scoreboard_prompt
    assert "Protocol:" not in scoreboard_prompt
    assert "Valid sampled frame indices" not in scoreboard_prompt
    assert "There are" not in scoreboard_prompt

    assert "These sampled inputs correspond to approximately" in actions_prompt
    assert "Valid sampled frame indices are" in actions_prompt
    assert "There are " in actions_prompt
    assert "question-relevant interval in sampled-frame indices is [20, 30]" in actions_prompt
    assert "Frames outside it are background/history context only." in actions_prompt


def test_render_prompt_for_objects_spatial_uses_required_labels():
    sample = PreparedSample(
        sample_id="video/sample#1",
        annotation_id="1",
        video_key="video/sample",
        task_name="Objects_Spatial_Relationships",
        task_level="independent",
        protocol_id="main",
        question_text="What is the spatial relationship between Player A and Player B?",
        sampled_frames_original=[0],
        sampled_to_original={0: 0},
        frame_files=["frames/0000.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "text": "Player A is to the right of Player B.",
            "objects": [
                {"label": "Player A", "bbox": [10, 20, 30, 40]},
                {"label": "Player B", "bbox": [50, 60, 70, 80]},
            ],
        },
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "one sampled video frame" in prompt
    assert "camera viewpoint" in prompt
    assert "`left` / `right`" in prompt
    assert "`in front of` / `behind`" in prompt
    assert "not who is leading in the play" in prompt
    assert "normalized_1000 coordinate system" in prompt
    assert '["Player A", "Player B"]' in prompt
    assert "The order of entries in `objects` does not matter." in prompt


def test_render_prompt_for_scoreboard_single_uses_whole_scoreboard_box():
    sample = PreparedSample(
        sample_id="video/sample#0",
        annotation_id="0",
        video_key="video/sample",
        task_name="Scoreboard_Single",
        task_level="independent",
        protocol_id="main",
        question_text="What is the current score?",
        sampled_frames_original=[0],
        sampled_to_original={0: 0},
        frame_files=["frames/0000.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "1-0", "bbox": [10, 20, 30, 40]},
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "one sampled video frame" in prompt
    assert "entire scoreboard" in prompt
    assert "full scoreboard" in prompt
    assert "normalized_1000 corner coordinates" in prompt
    assert "scoreboard region" not in prompt


def test_render_prompt_for_ai_coach_focuses_on_player_mistakes():
    sample = PreparedSample(
        sample_id="video/sample#2",
        annotation_id="2",
        video_key="video/sample",
        task_name="AI_Coach",
        task_level="independent",
        protocol_id="main",
        question_text="What did the player do wrong on this play?",
        sampled_frames_original=[0, 1, 2],
        sampled_to_original={0: 0, 1: 1, 2: 2},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "The player exposed the ball and lost balance."},
        sampled_video_fps=10.0,
        q_window_sampled=(0, 2),
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "Identify the player's mistakes" in prompt
    assert "approximately 10 fps" in prompt
    assert "actual mistakes" in prompt
    assert "improvement suggestions" in prompt
    assert "question-relevant interval in sampled-frame indices is [0, 2]" in prompt
    assert "reference answer" not in prompt
    assert "coaching advice" not in prompt


def test_render_prompt_for_score_prediction_uses_general_game_state_wording():
    sample = PreparedSample(
        sample_id="video/sample#4",
        annotation_id="4",
        video_key="video/sample",
        task_name="Score_Prediction",
        task_level="independent",
        protocol_id="main",
        question_text="Who is more likely to finish ahead based on the current standings?",
        sampled_frames_original=[0, 1, 2],
        sampled_to_original={0: 0, 1: 1, 2: 2},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "Team A is more likely to finish ahead."},
        sampled_video_fps=10.0,
        q_window_sampled=(0, 2),
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "ranking, score, and other visible game-state information" in prompt
    assert "approximately 10 fps" in prompt
    assert "question-relevant interval in sampled-frame indices is [0, 2]" in prompt
    assert "future result" not in prompt
    assert "predicted future result" not in prompt


def test_render_prompt_for_temporal_causal_uses_result_reasoning_wording():
    sample = PreparedSample(
        sample_id="video/sample#4a",
        annotation_id="4a",
        video_key="video/sample",
        task_name="Temporal_Causal",
        task_level="independent",
        protocol_id="main",
        question_text="Why did Team A lose the match?",
        sampled_frames_original=[0, 1, 2],
        sampled_to_original={0: 0, 1: 1, 2: 2},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "Team A conceded twice late in the game."},
        sampled_video_fps=10.0,
        q_window_sampled=(0, 2),
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "answer why the asked result happened" in prompt
    assert "approximately 10 fps" in prompt
    assert "win, loss, ranking, lead change, failure, or another competition outcome" in prompt
    assert "not merely restate the result" in prompt
    assert "main cause of the asked result" in prompt
    assert "question-relevant interval in sampled-frame indices is [0, 2]" in prompt


def test_render_prompt_for_spatial_imagination_describes_viewpoint_based_spatial_reasoning():
    sample = PreparedSample(
        sample_id="video/sample#3",
        annotation_id="3",
        video_key="video/sample",
        task_name="Spatial_Imagination",
        task_level="chain_downstream",
        protocol_id="main",
        question_text="Based on the previous play, where will the attacker move next?",
        sampled_frames_original=[0, 1, 2],
        sampled_to_original={0: 0, 1: 1, 2: 2},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={"text": "The attacker will cut toward the basket."},
        sampled_video_fps=10.0,
        q_window_sampled=(0, 2),
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "previous question-answer messages" in prompt
    assert "approximately 10 fps" in prompt
    assert "spatial conclusion required by that question" in prompt
    assert "specified viewpoint, observer position, or imagined camera angle" in prompt
    assert "position, movement trajectory, spatial relation, formation" in prompt
    assert "from the viewpoint requested in the question" in prompt
    assert "question-relevant interval in sampled-frame indices is [0, 2]" in prompt


def test_render_prompt_for_stg_uses_target_description_wording():
    sample = PreparedSample(
        sample_id="video/sample#5",
        annotation_id="5",
        video_key="video/sample",
        task_name="Spatial_Temporal_Grounding",
        task_level="independent",
        protocol_id="main",
        question_text="The athlete performs a jump.",
        sampled_frames_original=[100, 110, 120, 130],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130},
        frame_files=[
            "frames/0100.jpg",
            "frames/0110.jpg",
            "frames/0120.jpg",
            "frames/0130.jpg",
        ],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={},
        sampled_video_fps=3.0,
    )

    prompt = render_prompt(load_prompt_pack(PROMPT_ROOT), sample).prompt_text

    assert "Target Description:" in prompt
    assert "approximately 3 fps" in prompt
    assert "action or event involving a particular subject" in prompt
    assert "find when the described action or event happens" in prompt
    assert "track the subject referred to by that target description" in prompt
    assert "question-relevant interval" not in prompt
    assert "Question:" not in prompt


def test_render_oracle_prompt_uses_known_positions_wording(monkeypatch, tmp_path):
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
    actions_sample = next(
        sample
        for sample in load_prepared_samples(prepared_root, "main")
        if sample.task_name == "Continuous_Actions_Caption"
    )

    prompt = render_oracle_upstream_prompt(
        load_oracle_prompt_pack(ORACLE_PROMPT_ROOT),
        actions_sample,
        variant="language",
    ).prompt_text

    assert "OracleTrack" not in prompt
    assert "The target athlete's position is already known in some sampled frames." in prompt
    assert "These sampled inputs correspond to approximately" in prompt
    assert "normalized_1000 coordinate system" in prompt
    assert "The task is still to describe that target athlete's actions over time" in prompt
    assert "You do not need to output tracking boxes." in prompt
    assert "question-relevant interval in sampled-frame indices is [20, 30]" in prompt


def test_render_oracle_prompt_for_stg_uses_target_description_wording():
    sample = PreparedSample(
        sample_id="video/sample#6",
        annotation_id="6",
        video_key="video/sample",
        task_name="Spatial_Temporal_Grounding",
        task_level="independent",
        protocol_id="main",
        question_text="The athlete performs a jump.",
        sampled_frames_original=[100, 110, 120, 130],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130},
        frame_files=[
            "frames/0100.jpg",
            "frames/0110.jpg",
            "frames/0120.jpg",
            "frames/0130.jpg",
        ],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "tracking_gt_sampled": [
                {"frame_sampled": 0, "bbox_mot": [10, 20, 30, 40]},
                {"frame_sampled": 2, "bbox_mot": [12, 22, 30, 40]},
            ]
        },
        sampled_video_fps=3.0,
    )

    prompt = render_oracle_upstream_prompt(
        load_oracle_prompt_pack(ORACLE_PROMPT_ROOT),
        sample,
        variant="language",
    ).prompt_text

    assert "Target Description:" in prompt
    assert "approximately 3 fps" in prompt
    assert "action or event involving a particular subject" in prompt
    assert "find when the described action or event happens" in prompt
    assert "identify the subject referred to by the target description" in prompt
    assert "normalized_1000 coordinate system" in prompt
    assert "You do not need to output tracking boxes." in prompt
    assert "OracleTrack" not in prompt
    assert "question-relevant interval" not in prompt
    assert "Question:" not in prompt


def test_render_oracle_visual_prompt_uses_highlight_wording_without_tracking_json():
    sample = PreparedSample(
        sample_id="video/sample#7",
        annotation_id="7",
        video_key="video/sample",
        task_name="Continuous_Actions_Caption",
        task_level="independent",
        protocol_id="main",
        question_text="Describe the runner's actions.",
        sampled_frames_original=[0, 1, 2],
        sampled_to_original={0: 0, 1: 1, 2: 2},
        frame_files=["frames/0000.jpg", "frames/0001.jpg", "frames/0002.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "tracking_gt_sampled": [
                {"frame_sampled": 0, "bbox_mot": [10, 20, 30, 40]},
            ]
        },
        sampled_video_fps=10.0,
        q_window_sampled=(0, 2),
    )

    prompt = render_oracle_upstream_prompt(
        load_oracle_prompt_pack(ORACLE_PROMPT_ROOT),
        sample,
        variant="visual",
    ).prompt_text

    assert "highlighted with GT tracking boxes directly on the sampled inputs" in prompt
    assert "question-relevant interval in sampled-frame indices is [0, 2]" in prompt
    assert '"bbox_mot"' not in prompt
    assert "```json" not in prompt
    assert "You do not need to output tracking boxes." in prompt


def test_render_oracle_language_visual_prompt_combines_both_injections():
    sample = PreparedSample(
        sample_id="video/sample#8",
        annotation_id="8",
        video_key="video/sample",
        task_name="Spatial_Temporal_Grounding",
        task_level="independent",
        protocol_id="main",
        question_text="The athlete performs a jump.",
        sampled_frames_original=[100, 110, 120, 130],
        sampled_to_original={0: 100, 1: 110, 2: 120, 3: 130},
        frame_files=["frames/0100.jpg", "frames/0110.jpg", "frames/0120.jpg", "frames/0130.jpg"],
        source_video_path="video.mp4",
        source_annotation_path="anno.json",
        reference_payload={
            "tracking_gt_sampled": [
                {"frame_sampled": 0, "bbox_mot": [10, 20, 30, 40]},
            ]
        },
        sampled_video_fps=3.0,
    )

    prompt = render_oracle_upstream_prompt(
        load_oracle_prompt_pack(ORACLE_PROMPT_ROOT),
        sample,
        variant="language_visual",
    ).prompt_text

    assert "The target subject's position is already known in some sampled frames." in prompt
    assert "highlighted with GT tracking boxes directly on the sampled inputs" in prompt
    assert "question-relevant interval" not in prompt
    assert '"bbox_mot"' in prompt
