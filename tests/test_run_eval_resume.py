import json
from pathlib import Path

from PIL import Image

from omnichain_eval.adapters.base import BaseModelAdapter, MockAdapter
from omnichain_eval.cli import main
from omnichain_eval.judge import JudgeClient, JudgeResponseFormatExhaustedError, StaticJudgeClient
from omnichain_eval.metrics import evaluate_sample
from omnichain_eval.prepare import build_prepared_data, load_prepared_samples
from omnichain_eval.utils import write_jsonl


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mini_data"


def fake_decode_selected_frames(video_path: Path, frame_indices: list[int]):
    return {
        frame_index: Image.new("RGB", (64, 64), color=(frame_index % 255, 10, 20))
        for frame_index in frame_indices
    }


class AlwaysBrokenJudge(JudgeClient):
    def judge(
        self,
        task_name: str,
        prompt_text: str,
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

    @property
    def name(self) -> str:
        return "counting-mock"

    def supports_commentary(self) -> bool:
        return True

    def predict(self, sample, *, oracle_track: bool = False, context: dict | None = None):
        self.calls.append(sample.sample_id)
        return self._mock.predict(sample, oracle_track=oracle_track, context=context)


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
            MockAdapter().predict(sample),
            judge_client=AlwaysBrokenJudge(),
        )
    except JudgeResponseFormatExhaustedError as exc:
        assert str(exc) == "judge response did not match schema after 2 attempt(s)"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected JudgeResponseFormatExhaustedError")


def test_run_eval_resumes_from_existing_sample_results(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    prepared_samples = load_prepared_samples(prepared_root, "main")
    adapter = CountingAdapter()

    run_dir = artifacts_root / "resume-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    first_sample = prepared_samples[0]
    first_raw_output = adapter._mock.predict(first_sample)
    first_record = evaluate_sample(
        first_sample,
        first_raw_output,
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
    write_jsonl(run_dir / "sample_results.jsonl", [first_record.to_dict()])

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        f"""
[run_eval]
prepared_root = "{prepared_root}"
protocol = "main"
artifacts_root = "{artifacts_root}"
run_name = "resume-run"
adapter = "mock"

[judge]
backend = "static-pass"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: adapter)

    exit_code = main(["run-eval", "--config", str(config_path)])

    assert exit_code == 0
    assert first_sample.sample_id not in adapter.calls
    assert len(adapter.calls) == len(prepared_samples) - 1

    sample_result_rows = (run_dir / "sample_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    prediction_rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_result_rows) == len(prepared_samples)
    assert len(prediction_rows) == len(prepared_samples)


def test_run_eval_retries_predicted_but_not_evaluated_samples_on_next_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnichain_eval.prepare.decode_selected_frames",
        fake_decode_selected_frames,
    )
    prepared_root = tmp_path / "prepared"
    artifacts_root = tmp_path / "artifacts" / "runs"
    build_prepared_data(FIXTURE_ROOT, prepared_root, ["main"])
    adapter = CountingAdapter()

    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        f"""
[run_eval]
prepared_root = "{prepared_root}"
protocol = "main"
artifacts_root = "{artifacts_root}"
run_name = "retry-run"
adapter = "mock"

[judge]
backend = "static-pass"
invalid_json_retries = 1
""".strip(),
        encoding="utf-8",
    )

    class BrokenOnceJudge(JudgeClient):
        def __init__(self) -> None:
            self.calls = 0

        def judge(
            self,
            task_name: str,
            prompt_text: str,
            reference_payload: dict,
            prediction_payload: dict,
        ):
            self.calls += 1
            if self.calls == 1:
                raise JudgeResponseFormatExhaustedError("judge response did not match schema after 2 attempt(s)")
            return StaticJudgeClient(always_pass=True).judge(
                task_name,
                prompt_text,
                reference_payload,
                prediction_payload,
            )

    monkeypatch.setattr("omnichain_eval.cli.resolve_adapter", lambda spec: adapter)
    monkeypatch.setattr("omnichain_eval.cli._judge_client_from_config", lambda config: BrokenOnceJudge())

    first_exit_code = main(["run-eval", "--config", str(config_path)])

    assert first_exit_code == 0
    run_dir = artifacts_root / "retry-run"
    first_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert first_status["predicted_not_evaluated_samples_total"] == 1
    pending_ids = first_status["predicted_not_evaluated_sample_ids"]
    assert len(pending_ids) == 1

    adapter.calls.clear()
    monkeypatch.setattr(
        "omnichain_eval.cli._judge_client_from_config",
        lambda config: StaticJudgeClient(always_pass=True),
    )

    second_exit_code = main(["run-eval", "--config", str(config_path)])

    assert second_exit_code == 0
    assert adapter.calls == []
    second_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert second_status["predicted_not_evaluated_samples_total"] == 0
    sample_result_rows = (run_dir / "sample_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    prediction_rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(sample_result_rows) == len(load_prepared_samples(prepared_root, "main"))
    assert len(prediction_rows) == len(load_prepared_samples(prepared_root, "main"))
