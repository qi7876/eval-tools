import json
import shutil
from pathlib import Path
from textwrap import dedent

import pytest
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


def write_custom_protocol_module(tmp_path: Path) -> str:
    module_name = "custom_protocols_workflow"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from omnichain_eval.constants import SINGLE_FRAME_TASKS, TASK_STG
            from omnichain_eval.protocols import BaseProtocol, uniform_sample_closed_interval
            from omnichain_eval.utils import clip_index


            def _clip_closed_interval(start: int, end: int, total_frames: int) -> tuple[int, int]:
                clipped_start = clip_index(start, total_frames)
                clipped_end = clip_index(end, total_frames)
                if clipped_start > clipped_end:
                    clipped_start, clipped_end = clipped_end, clipped_start
                return clipped_start, clipped_end


            class EightFrameUniformProtocol(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Eight-frame uniform sampling."

                def sample_frames(self, sample):
                    total_frames = sample.video_metadata.total_frames
                    if sample.task_name in SINGLE_FRAME_TASKS:
                        if sample.timestamp_frame is None:
                            raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
                        return [sample.timestamp_frame]
                    if sample.task_name == TASK_STG:
                        if sample.a_window is None:
                            raise ValueError(f"{sample.sample_id} is missing A_window_frame")
                        start, end = _clip_closed_interval(
                            sample.a_window[0],
                            sample.a_window[1],
                            total_frames,
                        )
                        return uniform_sample_closed_interval(start, end, 8)
                    if sample.q_window is None:
                        raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
                    start, end = _clip_closed_interval(
                        sample.q_window[0],
                        sample.q_window[1],
                        total_frames,
                    )
                    return uniform_sample_closed_interval(start, end, 8)

                def to_manifest_dict(self):
                    return {
                        **super().to_manifest_dict(),
                        "sampling": "uniform",
                        "frame_budget": 8,
                    }


            class EightFrameUniformProtocolAlt(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Alternate eight-frame uniform sampling."

                def sample_frames(self, sample):
                    total_frames = sample.video_metadata.total_frames
                    if sample.task_name in SINGLE_FRAME_TASKS:
                        if sample.timestamp_frame is None:
                            raise ValueError(f"{sample.sample_id} is missing timestamp_frame")
                        return [sample.timestamp_frame]
                    if sample.task_name == TASK_STG:
                        if sample.a_window is None:
                            raise ValueError(f"{sample.sample_id} is missing A_window_frame")
                        start, end = _clip_closed_interval(
                            sample.a_window[0],
                            sample.a_window[1],
                            total_frames,
                        )
                        return uniform_sample_closed_interval(start, end, 4)
                    if sample.q_window is None:
                        raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
                    start, end = _clip_closed_interval(
                        sample.q_window[0],
                        sample.q_window[1],
                        total_frames,
                    )
                    return uniform_sample_closed_interval(start, end, 4)

                def to_manifest_dict(self):
                    return {
                        **super().to_manifest_dict(),
                        "sampling": "uniform",
                        "frame_budget": 4,
                    }
            """
        ),
        encoding="utf-8",
    )
    return module_name


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
        ["main"],
    )

    main_samples = load_prepared_samples(prepared_root, "main")
    assert len(main_samples) == 4

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
    assert chain_summary["base"]["chain_success"] == 1.0
    assert chain_summary["base"]["chain_success_wo_track"] == 1.0


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


def test_prepare_data_writes_sampled_query_interval(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])

    prepared_samples = load_prepared_samples(prepared_root, "main")
    actions_sample = next(sample for sample in prepared_samples if sample.annotation_id == "2")
    downstream_sample = next(sample for sample in prepared_samples if sample.annotation_id == "4")

    assert actions_sample.q_window == (100, 150)
    assert actions_sample.q_window_sampled == (20, 30)
    assert downstream_sample.q_window == (100, 150)
    assert downstream_sample.q_window_sampled == (20, 30)


def test_prepare_data_supports_custom_protocol_specs(monkeypatch, tmp_path):
    module_name = write_custom_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    protocol_spec = f"{module_name}:EightFrameUniformProtocol"
    prepared_root = tmp_path / "prepared"

    build_prepared_data(FIXTURE_ROOT, prepared_root, [protocol_spec])

    build_manifest = read_json(prepared_root / "native_uniform_8" / "build_manifest.json")
    assert build_manifest["protocol_id"] == "native_uniform_8"
    assert build_manifest["protocol_spec"] == protocol_spec
    assert build_manifest["protocol_manifest"]["frame_budget"] == 8

    prepared_samples = load_prepared_samples(prepared_root, protocol_spec)
    actions_sample = next(sample for sample in prepared_samples if sample.annotation_id == "2")
    downstream_sample = next(sample for sample in prepared_samples if sample.annotation_id == "4")

    assert len(actions_sample.sampled_frames_original) == 8
    assert actions_sample.q_window_sampled == (0, 7)
    assert downstream_sample.q_window_sampled == (0, 7)
    assert actions_sample.metadata["protocol_spec"] == protocol_spec


def test_load_prepared_samples_rejects_protocol_spec_mismatch(monkeypatch, tmp_path):
    module_name = write_custom_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    protocol_spec = f"{module_name}:EightFrameUniformProtocol"
    other_protocol_spec = f"{module_name}:EightFrameUniformProtocolAlt"

    build_prepared_data(FIXTURE_ROOT, prepared_root, [protocol_spec])

    with pytest.raises(ValueError, match="prepared protocol_spec"):
        load_prepared_samples(prepared_root, other_protocol_spec)


def test_prepare_data_writes_sampled_video_when_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )

    def fake_write_sampled_video(output_path: Path, frame_images, *, sampled_video_fps: float):
        assert frame_images
        output_path.write_bytes(b"fake mp4")
        assert sampled_video_fps > 0

    monkeypatch.setattr(
        "omnichain_eval.prepare._write_sampled_video",
        fake_write_sampled_video,
    )

    prepared_root = tmp_path / "prepared"
    build_prepared_data(
        FIXTURE_ROOT,
        prepared_root,
        ["main"],
        media_formats=["frames", "sampled_video"],
    )

    prepared_samples = load_prepared_samples(prepared_root, "main")
    multi_frame_sample = next(sample for sample in prepared_samples if len(sample.frame_files) > 1)
    single_frame_sample = next(sample for sample in prepared_samples if sample.task_name == "Scoreboard_Single")

    assert multi_frame_sample.sampled_video_file is not None
    assert Path(multi_frame_sample.sampled_video_file).is_absolute()
    assert Path(multi_frame_sample.sampled_video_file).exists()
    assert multi_frame_sample.sampled_video_fps is not None
    assert multi_frame_sample.sampled_video_fps > 0
    assert single_frame_sample.sampled_video_file is None
    assert single_frame_sample.sampled_video_fps is None

    build_manifest = read_json(prepared_root / "main" / "build_manifest.json")
    assert build_manifest["media_formats"] == ["frames", "sampled_video"]


def test_prepare_data_writes_oracle_visual_media_when_requested(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )

    def fake_write_sampled_video(output_path: Path, frame_images, *, sampled_video_fps: float):
        assert frame_images
        output_path.write_bytes(b"fake mp4")
        assert sampled_video_fps > 0

    monkeypatch.setattr(
        "omnichain_eval.prepare._write_sampled_video",
        fake_write_sampled_video,
    )

    prepared_root = tmp_path / "prepared"
    build_prepared_data(
        FIXTURE_ROOT,
        prepared_root,
        ["main"],
        media_formats=["frames", "sampled_video"],
        generate_oracle_visual_media=True,
    )

    prepared_samples = load_prepared_samples(prepared_root, "main")
    oracle_upstream_sample = next(
        sample for sample in prepared_samples if sample.task_name == "Continuous_Actions_Caption"
    )
    non_oracle_sample = next(
        sample for sample in prepared_samples if sample.task_name == "Spatial_Imagination"
    )

    assert oracle_upstream_sample.oracle_visual_frame_files
    assert all(Path(path).is_absolute() for path in oracle_upstream_sample.oracle_visual_frame_files)
    assert all(Path(path).exists() for path in oracle_upstream_sample.oracle_visual_frame_files)
    assert oracle_upstream_sample.oracle_visual_sampled_video_file is not None
    assert Path(oracle_upstream_sample.oracle_visual_sampled_video_file).is_absolute()
    assert Path(oracle_upstream_sample.oracle_visual_sampled_video_file).exists()
    assert non_oracle_sample.oracle_visual_frame_files == []
    assert non_oracle_sample.oracle_visual_sampled_video_file is None

    build_manifest = read_json(prepared_root / "main" / "build_manifest.json")
    assert build_manifest["generate_oracle_visual_media"] is True


def test_prepare_data_fails_loudly_without_pyav(monkeypatch, tmp_path):
    monkeypatch.setattr("omnichain_eval.prepare.av", None)

    with pytest.raises(RuntimeError, match="prepared-data video decoding/encoding requires PyAV"):
        build_prepared_data(
            FIXTURE_ROOT,
            tmp_path / "prepared",
            ["main"],
            media_formats=["frames"],
        )


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
        ),
        ChainPairRecord(
            pair_id="down#2|up#2",
            video_key="TestSport/TestEvent/1",
            upstream_sample_id="up#2",
            downstream_sample_id="down#2",
            upstream_task_name="Spatial_Temporal_Grounding",
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
        "up#2": _evaluation_record(
            "up#2",
            "Spatial_Temporal_Grounding",
            component_pass={"tracking_pass": 1, "tiou_pass": 1},
        ),
        "down#2": _evaluation_record(
            "down#2",
            "Spatial_Imagination",
            component_pass={"judge_pass": 0},
        ),
    }
    oracle_variant_pair_results = {
        "language": {
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
            },
            "down#2|up#2": {
                "upstream": _evaluation_record(
                    "up#2",
                    "Spatial_Temporal_Grounding",
                    component_pass={"tracking_pass": 1, "tiou_pass": 0},
                ),
                "downstream": _evaluation_record(
                    "down#2",
                    "Spatial_Imagination",
                    component_pass={"judge_pass": 1},
                ),
            },
        }
    }

    chain_summary = summarize_experiment_b(
        pairs,
        records_by_sample_id,
        oracle_variant_pair_results=oracle_variant_pair_results,
    )

    assert chain_summary["base"]["understanding_acc"] == 0.5
    assert chain_summary["base"]["understanding_acc_by_task"] == {
        "Continuous_Actions_Caption": 0.0,
        "Spatial_Temporal_Grounding": 1.0,
    }
    assert chain_summary["base"]["understanding_acc_wo_track"] == 1.0
    assert chain_summary["base"]["understanding_acc_wo_track_by_task"] == {
        "Continuous_Actions_Caption": 1.0,
        "Spatial_Temporal_Grounding": 1.0,
    }
    assert chain_summary["base"]["reasoning_acc"] == 0.5
    assert chain_summary["base"]["chain_success"] == 0.0
    assert chain_summary["base"]["chain_success_wo_track"] == 0.5
    assert "understanding_acc" not in chain_summary["oracle_language"]
    assert "understanding_acc_by_task" not in chain_summary["oracle_language"]
    assert chain_summary["oracle_language"]["understanding_acc_wo_track"] == 0.5
    assert chain_summary["oracle_language"]["understanding_acc_wo_track_by_task"] == {
        "Continuous_Actions_Caption": 1.0,
        "Spatial_Temporal_Grounding": 0.0,
    }
    assert chain_summary["oracle_language"]["reasoning_acc"] == 1.0
    assert chain_summary["oracle_language"]["chain_success_wo_track"] == 0.5
    assert chain_summary["oracle_visual"]["num_pending_chain_samples"] == 2
    assert chain_summary["oracle_language_visual"]["num_pending_chain_samples"] == 2


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
