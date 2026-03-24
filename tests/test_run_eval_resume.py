import json
from pathlib import Path

from PIL import Image

from omnichain_eval.adapters.base import BaseModelAdapter, MockAdapter
from omnichain_eval.cli import main
from omnichain_eval.experiments import build_chain_manifest
from omnichain_eval.judge import JudgeClient, JudgeResponseFormatExhaustedError, StaticJudgeClient
from omnichain_eval.metrics import evaluate_sample
from omnichain_eval.prepare import build_prepared_data, load_prepared_samples
from omnichain_eval.prompting import build_model_input, load_prompt_pack, render_prompt
from omnichain_eval.schema import ModelInput, StructuredPredictionRecord
from omnichain_eval.structurer import (
    StaticParseStructurerBackend,
    StructurerService,
    load_structurer_prompt_pack,
)
from omnichain_eval.utils import write_jsonl


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "benchmark_v1"
STRUCTURER_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts" / "structurer_v1"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_v1.md"


def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
    return {
        frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
        for frame_index in frame_indices
    }


def build_test_model_input(sample, *, oracle_track: bool = False) -> ModelInput:
    prompt_pack = load_prompt_pack(PROMPT_ROOT)
    return build_model_input(
        sample,
        render_prompt(prompt_pack, sample, oracle_track=oracle_track),
        oracle_track=oracle_track,
    )


def build_test_structured_record(sample, raw_output: str) -> StructuredPredictionRecord:
    structurer_service = StructurerService(
        backend=StaticParseStructurerBackend(),
        prompt_pack=load_structurer_prompt_pack(STRUCTURER_PROMPT_ROOT),
        invalid_json_retries=0,
    )
    structured_result = structurer_service.structure(sample, raw_output)
    return StructuredPredictionRecord(
        sample_id=sample.sample_id,
        task_name=sample.task_name,
        video_key=sample.video_key,
        protocol_id=sample.protocol_id,
        raw_output=structured_result.raw_output,
        structured_prediction=structured_result.structured_prediction,
        structuring_errors=structured_result.errors,
        structuring_warnings=structured_result.warnings,
        structurer_raw_response=structured_result.structurer_raw_response,
    )


class AlwaysBrokenJudge(JudgeClient):
    def judge(
        self,
        task_name: str,
        question_text: str,
        reference_payload: dict,
        prediction_payload: dict,
    ):
        raise JudgeResponseFormatExhaustedError(
            "judge response did not match schema after 2 attempt(s)"
        )


class CountingAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self._mock = MockAdapter()
        self.calls: list[str] = []
        self.inputs: dict[str, list[ModelInput]] = {}

    @property
    def name(self) -> str:
        return "counting-mock"

    def predict(self, model_input: ModelInput):
        sample_id = model_input.sample.sample_id
        self.calls.append(sample_id)
        self.inputs.setdefault(sample_id, []).append(model_input)
        return self._mock.predict(model_input)


def _config_text(
    *,
    prepared_root: Path,
    artifacts_root: Path,
    run_name: str,
    chain_manifest: Path | None,
    judge_invalid_json_retries: int = 0,
    structurer_invalid_json_retries: int = 0,
) -> str:
    chain_manifest_line = f'chain_manifest = "{chain_manifest}"\n' if chain_manifest else ""
    return f"""
[run_eval]
prepared_root = "{prepared_root}"
protocol = "main"
artifacts_root = "{artifacts_root}"
prompt_root = "{PROMPT_ROOT}"
run_name = "{run_name}"
adapter = "mock"
{chain_manifest_line}

[judge]
backend = "static-pass"
prompt_path = "{JUDGE_PROMPT_PATH}"
invalid_json_retries = {judge_invalid_json_retries}
concurrency = 1

[structurer]
backend = "static-parse"
prompt_root = "{STRUCTURER_PROMPT_ROOT}"
invalid_json_retries = {structurer_invalid_json_retries}
concurrency = 1
""".strip()


def test_evaluate_sample_skips_question_when_judge_format_errors_are_exhausted(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    sample = next(
        candidate
        for candidate in load_prepared_samples(prepared_root, "main")
        if candidate.task_name != "Spatial_Temporal_Grounding"
    )

    try:
        evaluate_sample(
            sample,
            build_test_structured_record(
                sample,
                MockAdapter().predict(build_test_model_input(sample)),
            ),
            judge_client=AlwaysBrokenJudge(),
        )
    except JudgeResponseFormatExhaustedError as exc:
        assert str(exc) == "judge response did not match schema after 2 attempt(s)"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected JudgeResponseFormatExhaustedError")


def test_run_eval_resumes_from_existing_results(monkeypatch, tmp_path):
    monkeypatch.setattr("omnichain_eval.experiments.compute_bertscore", lambda records: None)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    chain_manifest_path = tmp_path / "chain_pairs.jsonl"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    build_chain_manifest(FIXTURE_ROOT, chain_manifest_path)
    prepared_samples = load_prepared_samples(prepared_root, "main")
    adapter = CountingAdapter()

    run_dir = artifacts_root / "resume-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    first_sample = next(sample for sample in prepared_samples if sample.task_name != "Spatial_Imagination")
    first_raw_output = adapter._mock.predict(build_test_model_input(first_sample))
    first_structured_record = build_test_structured_record(first_sample, first_raw_output)
    first_record = evaluate_sample(
        first_sample,
        first_structured_record,
        judge_client=StaticJudgeClient(always_pass=True),
    )
    write_jsonl(
        run_dir / "predictions.jsonl",
        [
            {
                "sample_id": first_sample.sample_id,
                "raw_output": first_raw_output,
                "protocol_id": "main",
            }
        ],
    )
    write_jsonl(
        run_dir / "structured_predictions.jsonl",
        [first_structured_record.to_dict()],
    )
    write_jsonl(run_dir / "results.jsonl", [first_record.to_dict()])

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        _config_text(
            prepared_root=prepared_root,
            artifacts_root=artifacts_root,
            run_name="resume-run",
            chain_manifest=chain_manifest_path,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: adapter)

    exit_code = main(["run-eval", "--config", str(config_path)])

    assert exit_code == 0
    assert first_sample.sample_id not in adapter.calls
    assert len(adapter.calls) == len(prepared_samples) - 1

    sample_result_rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    structured_rows = (
        run_dir / "structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    prediction_rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chain_prediction_rows = (
        run_dir / "chain_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    chain_structured_rows = (
        run_dir / "chain_structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    chain_result_rows = (run_dir / "chain_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_result_rows) + len(chain_result_rows) == len(prepared_samples)
    assert len(prediction_rows) + len(chain_prediction_rows) == len(prepared_samples)
    assert len(structured_rows) + len(chain_structured_rows) == len(prepared_samples)


def test_run_eval_retries_predicted_but_not_evaluated_samples_on_next_run(monkeypatch, tmp_path):
    monkeypatch.setattr("omnichain_eval.experiments.compute_bertscore", lambda records: None)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    chain_manifest_path = tmp_path / "chain_pairs.jsonl"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    build_chain_manifest(FIXTURE_ROOT, chain_manifest_path)
    adapter = CountingAdapter()

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        _config_text(
            prepared_root=prepared_root,
            artifacts_root=artifacts_root,
            run_name="retry-run",
            chain_manifest=chain_manifest_path,
            judge_invalid_json_retries=1,
        ),
        encoding="utf-8",
    )

    class BrokenOnceJudge(JudgeClient):
        def __init__(self) -> None:
            self.calls = 0

        def judge(
            self,
            task_name: str,
            question_text: str,
            reference_payload: dict,
            prediction_payload: dict,
        ):
            self.calls += 1
            if self.calls == 1:
                raise JudgeResponseFormatExhaustedError(
                    "judge response did not match schema after 2 attempt(s)"
                )
            return StaticJudgeClient(always_pass=True).judge(
                task_name,
                question_text,
                reference_payload,
                prediction_payload,
            )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: adapter)
    monkeypatch.setattr("omnichain_eval.cli._judge_client_from_config", lambda config: BrokenOnceJudge())

    first_exit_code = main(["run-eval", "--config", str(config_path)])

    assert first_exit_code == 0
    run_dir = artifacts_root / "retry-run"
    first_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    first_status = first_summary["run_status"]
    pending_ids = first_status["structured_not_evaluated_sample_ids"]
    assert len(pending_ids) == 1

    adapter.calls.clear()
    monkeypatch.setattr(
        "omnichain_eval.cli._judge_client_from_config",
        lambda config: StaticJudgeClient(always_pass=True),
    )

    second_exit_code = main(["run-eval", "--config", str(config_path)])

    assert second_exit_code == 0
    assert adapter.calls == []
    second_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    second_status = second_summary["run_status"]
    assert second_status["structured_not_evaluated_sample_ids"] == []
    sample_result_rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    structured_rows = (
        run_dir / "structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    prediction_rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chain_prediction_rows = (
        run_dir / "chain_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    chain_structured_rows = (
        run_dir / "chain_structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    chain_result_rows = (run_dir / "chain_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_result_rows) + len(chain_result_rows) == len(load_prepared_samples(prepared_root, "main"))
    assert len(prediction_rows) + len(chain_prediction_rows) == len(load_prepared_samples(prepared_root, "main"))
    assert len(structured_rows) + len(chain_structured_rows) == len(load_prepared_samples(prepared_root, "main"))


def test_run_eval_splits_chain_outputs_and_passes_history_messages(monkeypatch, tmp_path):
    monkeypatch.setattr("omnichain_eval.experiments.compute_bertscore", lambda records: None)
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    chain_manifest_path = tmp_path / "chain_pairs.jsonl"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    build_chain_manifest(FIXTURE_ROOT, chain_manifest_path)
    adapter = CountingAdapter()

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        _config_text(
            prepared_root=prepared_root,
            artifacts_root=artifacts_root,
            run_name="chain-run",
            chain_manifest=chain_manifest_path,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: adapter)

    exit_code = main(["run-eval", "--config", str(config_path)])

    assert exit_code == 0
    run_dir = artifacts_root / "chain-run"
    prediction_rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    structured_rows = (
        run_dir / "structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    result_rows = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chain_prediction_rows = (run_dir / "chain_predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    chain_structured_rows = (
        run_dir / "chain_structured_predictions.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    chain_result_rows = (run_dir / "chain_results.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert len(prediction_rows) == 3
    assert len(structured_rows) == 3
    assert len(result_rows) == 3
    assert len(chain_prediction_rows) == 1
    assert len(chain_structured_rows) == 1
    assert len(chain_result_rows) == 1

    downstream_sample_id = "TestSport/TestEvent/1#4"
    downstream_inputs = adapter.inputs[downstream_sample_id]
    assert len(downstream_inputs) == 1
    model_input = downstream_inputs[0]
    messages = model_input.messages_as_dicts()
    upstream_sample = next(
        sample
        for sample in load_prepared_samples(prepared_root, "main")
        if sample.sample_id == "TestSport/TestEvent/1#2"
    )
    expected_upstream_output = MockAdapter().predict(build_test_model_input(upstream_sample))
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "Describe the athlete actions."
    assert messages[1]["content"] == expected_upstream_output

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"]["pending_chain_prediction_sample_ids"] == []
    assert summary["run_status"]["chain_predicted_not_structured_sample_ids"] == []
    assert summary["run_status"]["chain_structured_not_evaluated_sample_ids"] == []
    assert summary["run_status"]["blocked_chain_sample_ids"] == []


def test_run_eval_requires_chain_manifest_for_spatial_imagination(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        _config_text(
            prepared_root=prepared_root,
            artifacts_root=artifacts_root,
            run_name="missing-chain-manifest",
            chain_manifest=None,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: CountingAdapter())

    try:
        main(["run-eval", "--config", str(config_path)])
    except ValueError as exc:
        assert "chain_manifest" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected run-eval to fail without chain_manifest")
