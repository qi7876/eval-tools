from pathlib import Path

from PIL import Image

from omnichain_eval.adapters.base import MockAdapter
from omnichain_eval.dataset import scan_dataset
from omnichain_eval.experiments import (
    build_chain_manifest,
    evaluate_prepared_predictions,
    load_chain_pairs,
    summarize_experiment_b,
)
from omnichain_eval.judge import StaticJudgeClient
from omnichain_eval.prepare import build_prepared_data, load_prepared_samples
from omnichain_eval.prompting import build_model_input, load_prompt_pack, render_prompt


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"


def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
    return {
        frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
        for frame_index in frame_indices
    }


def test_end_to_end_prepare_and_eval(monkeypatch, tmp_path):
    records, issues = scan_dataset(FIXTURE_ROOT)
    assert not issues
    assert len(records) == 5

    chain_path = tmp_path / "chain_pairs.jsonl"
    pairs = build_chain_manifest(FIXTURE_ROOT, chain_path)
    assert len(pairs) == 1

    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(
        FIXTURE_ROOT,
        prepared_root,
        ["main", "expd_window_32s_2fps"],
    )

    main_samples = load_prepared_samples(prepared_root, "main")
    expd_samples = load_prepared_samples(prepared_root, "expd_window_32s_2fps")
    assert len(main_samples) == 5
    assert {sample.task_name for sample in expd_samples} == {
        "Continuous_Actions_Caption",
        "Spatial_Imagination",
        "Commentary",
    }

    adapter = MockAdapter()
    prompt_pack = load_prompt_pack(PROMPT_ROOT)
    prediction_map = {
        sample.sample_id: adapter.predict(
            build_model_input(sample, render_prompt(prompt_pack, sample))
        )
        for sample in main_samples
    }
    evaluation = evaluate_prepared_predictions(
        main_samples,
        prediction_map,
        model_name="mock",
        judge_client=StaticJudgeClient(always_pass=True),
        commentary_supported=True,
    )
    assert evaluation["overall"] == 1.0

    pairs = load_chain_pairs(chain_path)
    chain_summary = summarize_experiment_b(pairs, evaluation["records_by_sample_id"])
    assert chain_summary["chain_success"] == 1.0
