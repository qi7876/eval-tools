from pathlib import Path

import pytest

from omnichain_eval.cli import main
from omnichain_eval.config import load_run_eval_config


def test_load_run_eval_config_resolves_paths_and_judge_options(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run_eval.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "../prepared_data"
protocol = "main"
artifacts_root = "../artifacts/runs"
run_name = "demo-run"
model_name = "demo-model"
adapter = "mock"
chain_manifest = "../artifacts/chain_pairs.jsonl"
enable_oracle_track = true

[judge]
backend = "openai"
base_url = "http://judge.example/v1"
api_key_env = "CUSTOM_JUDGE_KEY"
model = "judge-model"
temperature = 0.2
top_p = 0.95
top_k = 4
max_tokens = 512
n = 2
seed = 7
invalid_json_retries = 3
""".strip(),
        encoding="utf-8",
    )

    config = load_run_eval_config(config_path)

    assert config.prepared_root == (tmp_path / "prepared_data").resolve()
    assert config.protocol == "main"
    assert config.artifacts_root == (tmp_path / "artifacts" / "runs").resolve()
    assert config.run_name == "demo-run"
    assert config.model_name == "demo-model"
    assert config.adapter == "mock"
    assert config.chain_manifest == (tmp_path / "artifacts" / "chain_pairs.jsonl").resolve()
    assert config.enable_oracle_track is True
    assert config.judge.backend == "openai"
    assert config.judge.base_url == "http://judge.example/v1"
    assert config.judge.api_key_env == "CUSTOM_JUDGE_KEY"
    assert config.judge.model == "judge-model"
    assert config.judge.temperature == 0.2
    assert config.judge.top_p == 0.95
    assert config.judge.top_k == 4
    assert config.judge.max_tokens == 512
    assert config.judge.n == 2
    assert config.judge.seed == 7
    assert config.judge.invalid_json_retries == 3


def test_load_run_eval_config_requires_exactly_one_source(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one of adapter or predictions"):
        load_run_eval_config(config_path)


def test_prepare_data_command_uses_toml_config(tmp_path, monkeypatch):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main", "expd_window_32s_2fps"]
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_prepared_data(data_root: Path, prepared_root: Path, protocols: list[str]):
        captured["data_root"] = data_root
        captured["prepared_root"] = prepared_root
        captured["protocols"] = protocols
        return [
            {
                "num_prepared_samples": 3,
                "protocol_id": "main",
                "protocol_root": str(prepared_root / "main"),
                "dataset_issue_count": 0,
            },
            {
                "num_prepared_samples": 2,
                "protocol_id": "expd_window_32s_2fps",
                "protocol_root": str(prepared_root / "expd_window_32s_2fps"),
                "dataset_issue_count": 0,
            },
        ]

    monkeypatch.setattr("omnichain_eval.cli.build_prepared_data", fake_build_prepared_data)

    exit_code = main(["prepare-data", "--config", str(config_path)])

    assert exit_code == 0
    assert captured["data_root"] == (tmp_path / "data").resolve()
    assert captured["prepared_root"] == (tmp_path / "prepared_data").resolve()
    assert captured["protocols"] == ["main", "expd_window_32s_2fps"]
