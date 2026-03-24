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
from omnichain_eval.schema import StructuredPredictionRecord
from omnichain_eval.structurer import (
    StaticParseStructurerBackend,
    StructurerService,
    load_structurer_prompt_pack,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"
STRUCTURER_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "structurer_v1"


def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
    return {
        frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
        for frame_index in frame_indices
    }


def test_end_to_end_prepare_and_eval(monkeypatch, tmp_path):
    monkeypatch.setattr("omnichain_eval.experiments.compute_bertscore", lambda records: None)
    records, issues = scan_dataset(FIXTURE_ROOT)
    assert not issues
    assert len(records) == 4

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
    assert len(main_samples) == 4
    assert {sample.task_name for sample in expd_samples} == {
        "Continuous_Actions_Caption",
        "Spatial_Imagination",
    }

    adapter = MockAdapter()
    prompt_pack = load_prompt_pack(PROMPT_ROOT)
    structurer_service = StructurerService(
        backend=StaticParseStructurerBackend(),
        prompt_pack=load_structurer_prompt_pack(STRUCTURER_PROMPT_ROOT),
        invalid_json_retries=0,
    )
    structured_prediction_map = {}
    for sample in main_samples:
        raw_output = adapter.predict(build_model_input(sample, render_prompt(prompt_pack, sample)))
        structured_result = structurer_service.structure(sample, raw_output)
        structured_prediction_map[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "task_name": sample.task_name,
            "video_key": sample.video_key,
            "protocol_id": sample.protocol_id,
            "raw_output": structured_result.raw_output,
            "structured_prediction": structured_result.structured_prediction,
            "structuring_errors": structured_result.errors,
            "structuring_warnings": structured_result.warnings,
            "structurer_raw_response": structured_result.structurer_raw_response,
        }
    evaluation = evaluate_prepared_predictions(
        main_samples,
        {
            sample_id: StructuredPredictionRecord.from_dict(payload)
            for sample_id, payload in structured_prediction_map.items()
        },
        model_name="mock",
        judge_client=StaticJudgeClient(always_pass=True),
    )
    assert evaluation["overall"] == 1.0

    pairs = load_chain_pairs(chain_path)
    chain_summary = summarize_experiment_b(pairs, evaluation["records_by_sample_id"])
    assert chain_summary["chain_success"] == 1.0
