import json
import shutil
from pathlib import Path

from PIL import Image

from omnichain_eval.cli import main
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
from omnichain_eval.schema import ChainPairRecord, EvaluationRecord, StructuredPredictionRecord
from omnichain_eval.structurer import (
    StaticParseStructurerBackend,
    StructurerService,
    load_structurer_prompt_pack,
)
from omnichain_eval.utils import read_json, read_jsonl


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"
STRUCTURER_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "structurer_v1"


def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
    return {
        frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
        for frame_index in frame_indices
    }


def build_fixture_with_unsupported_task(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    shutil.copytree(FIXTURE_ROOT, dataset_root)
    annotation_path = dataset_root / "TestSport" / "TestEvent" / "1.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"].append(
        {
            "annotation_id": "5",
            "task_L1": "Understanding",
            "task_L2": "Commentary",
            "Q_window_frame": [0, 20],
            "question": "Provide live commentary.",
            "answer": "The player keeps moving forward.",
        }
    )
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_root


def build_fixture_with_q_window_end_equal_total_frames(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    shutil.copytree(FIXTURE_ROOT, dataset_root)
    annotation_path = dataset_root / "TestSport" / "TestEvent" / "1.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    for annotation in payload["annotations"]:
        if annotation["annotation_id"] == "2":
            annotation["Q_window_frame"] = [100, payload["video_metadata"]["total_frames"]]
            break
    annotation_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_root


def build_fixture_with_two_videos(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    shutil.copytree(FIXTURE_ROOT, dataset_root)
    source_dir = dataset_root / "TestSport" / "TestEvent"
    copied_dir = dataset_root / "TestSport" / "TestEventParallel"
    shutil.copytree(source_dir, copied_dir)
    return dataset_root


def _evaluation_record(
    sample_id: str,
    task_name: str,
    *,
    component_pass: dict[str, int],
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task_name=task_name,
        video_key="TestSport/TestEvent/1",
        protocol_id="main",
        structured_prediction={},
        structuring_errors=[],
        structuring_warnings=[],
        component_metrics={},
        component_pass=component_pass,
        task_pass=1,
        raw_output="{}",
    )


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
    assert chain_summary["chain_success_wo_track"] == 1.0


def test_prepare_data_clips_q_window_end_equal_to_total_frames(monkeypatch, tmp_path):
    dataset_root = build_fixture_with_q_window_end_equal_total_frames(tmp_path)
    requested_frame_indices: list[int] = []

    def recording_decode(video_path: Path, frame_indices: list[int]):
        del video_path
        requested_frame_indices.extend(frame_indices)
        assert max(frame_indices) <= 499
        return fake_decode_selected_frames(Path("unused.mp4"), frame_indices)

    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        recording_decode,
    )

    prepared_root = tmp_path / "prepared"
    build_prepared_data(dataset_root, prepared_root, ["main"])

    assert requested_frame_indices
    assert 500 not in requested_frame_indices

    prepared_samples = load_prepared_samples(prepared_root, "main")
    long_window_sample = next(sample for sample in prepared_samples if sample.annotation_id == "2")
    assert long_window_sample.sampled_frames_original[-1] == 499
    assert len(long_window_sample.sampled_frames_original) == 64


def test_prepare_data_parallel_workers_preserve_deterministic_outputs(monkeypatch, tmp_path):
    dataset_root = build_fixture_with_two_videos(tmp_path)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )

    prepared_root_serial = tmp_path / "prepared_serial"
    prepared_root_parallel = tmp_path / "prepared_parallel"

    build_prepared_data(dataset_root, prepared_root_serial, ["main"], workers=1)
    build_prepared_data(dataset_root, prepared_root_parallel, ["main"], workers=2)

    assert read_jsonl(prepared_root_serial / "main" / "index.jsonl") == read_jsonl(
        prepared_root_parallel / "main" / "index.jsonl"
    )
    assert read_json(prepared_root_serial / "main" / "stats.json") == read_json(
        prepared_root_parallel / "main" / "stats.json"
    )
    assert read_json(
        prepared_root_serial / "main" / "build_manifest.json"
    ) == read_json(prepared_root_parallel / "main" / "build_manifest.json")


def test_prepare_data_writes_normalized_coordinate_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])

    build_manifest = read_json(prepared_root / "main" / "build_manifest.json")
    assert build_manifest["coordinate_system"] == "normalized_1000"

    prepared_sample = load_prepared_samples(prepared_root, "main")[0]
    assert prepared_sample.metadata["coordinate_system"] == "normalized_1000"
    assert prepared_sample.metadata["frame_width"] > 0
    assert prepared_sample.metadata["frame_height"] > 0


def test_load_prepared_samples_rejects_legacy_coordinate_system(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])

    build_manifest_path = prepared_root / "main" / "build_manifest.json"
    build_manifest = read_json(build_manifest_path)
    build_manifest["coordinate_system"] = "pixel"
    build_manifest_path.write_text(json.dumps(build_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        load_prepared_samples(prepared_root, "main")
    except ValueError as exc:
        assert "unsupported prepared coordinate_system" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected load_prepared_samples to reject legacy coordinate system")


def test_summarize_experiment_b_tracks_wo_track_metrics_and_oracle_names():
    pairs = [
        ChainPairRecord(
            pair_id="down#1|up#1",
            video_key="TestSport/TestEvent/1",
            upstream_sample_id="up#1",
            downstream_sample_id="down#1",
            upstream_task_name="Continuous_Actions_Caption",
        )
    ]
    records_by_sample_id = {
        "up#1": _evaluation_record(
            "up#1",
            "Continuous_Actions_Caption",
            component_pass={"tracking_pass": 0, "judge_pass": 1},
        ),
        "down#1": _evaluation_record(
            "down#1",
            "Spatial_Imagination",
            component_pass={"judge_pass": 1},
        ),
    }
    oracle_pair_results = {
        "down#1|up#1": {
            "upstream": _evaluation_record(
                "up#1",
                "Continuous_Actions_Caption",
                component_pass={"tracking_pass": 1, "judge_pass": 1},
            ),
            "downstream": _evaluation_record(
                "down#1",
                "Spatial_Imagination",
                component_pass={"judge_pass": 1},
            ),
        }
    }

    chain_summary = summarize_experiment_b(
        pairs,
        records_by_sample_id,
        oracle_pair_results=oracle_pair_results,
    )

    assert chain_summary["understanding_acc"] == 0.0
    assert chain_summary["reasoning_acc"] == 1.0
    assert chain_summary["chain_success"] == 0.0
    assert chain_summary["chain_success_wo_track"] == 1.0
    assert chain_summary["understanding_acc_oracle"] == 1.0
    assert chain_summary["reasoning_acc_oracle"] == 1.0
    assert chain_summary["chain_success_wo_track_oracle"] == 1.0
    assert "chain_success_oracle" not in chain_summary


def test_unsupported_tasks_are_ignored_in_validation_prepare_and_chain(monkeypatch, tmp_path):
    dataset_root = build_fixture_with_unsupported_task(tmp_path)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )

    records, issues = scan_dataset(dataset_root)
    assert not issues
    assert len(records) == 4

    validate_config = tmp_path / "validate.toml"
    validate_config.write_text(
        f"""
[validate_data]
data_root = "{dataset_root}"
""".strip(),
        encoding="utf-8",
    )
    assert main(["validate-data", "--config", str(validate_config)]) == 0

    chain_path = tmp_path / "chain_pairs.jsonl"
    pairs = build_chain_manifest(dataset_root, chain_path)
    assert len(pairs) == 1

    prepared_root = tmp_path / "prepared"
    build_prepared_data(dataset_root, prepared_root, ["main"])
    prepared_samples = load_prepared_samples(prepared_root, "main")
    assert len(prepared_samples) == 4

    build_manifest = read_json(prepared_root / "main" / "build_manifest.json")
    assert build_manifest["data_status"]["ignored_unsupported_sample_count"] == 1
    assert build_manifest["data_status"]["ignored_unsupported_task_counts"] == {"Commentary": 1}

    stats = read_json(prepared_root / "main" / "stats.json")
    assert stats["ignored_unsupported_sample_count"] == 1
    assert stats["ignored_unsupported_task_counts"] == {"Commentary": 1}
