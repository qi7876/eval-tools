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
prompt_root = "../prompts/benchmark_v1"
run_name = "demo-run"
model_name = "demo-model"
adapter = "mock"
chain_manifest = "../artifacts/chain_pairs.jsonl"
enable_oracle_track = true
oracle_prompt_root = "../prompts/benchmark_oracle_v1"

[judge]
backend = "openai"
prompt_root = "../prompts/judge_v1"
base_url = "http://judge.example/v1"
api_key_env = "CUSTOM_JUDGE_KEY"
model = "judge-model"
invalid_json_retries = 3
concurrency = 2

[judge.extra_body.thinking]
type = "disabled"

[structurer]
backend = "openai"
prompt_root = "../prompts/structurer_v1"
oracle_prompt_root = "../prompts/structurer_oracle_v1"
base_url = "http://structurer.example/v1"
api_key_env = "CUSTOM_STRUCTURER_KEY"
model = "structurer-model"
invalid_json_retries = 4
concurrency = 5

[structurer.extra_body.thinking]
type = "disabled"
""".strip(),
        encoding="utf-8",
    )

    config = load_run_eval_config(config_path)

    assert config.prepared_root == (tmp_path / "prepared_data").resolve()
    assert config.protocol == "main"
    assert config.artifacts_root == (tmp_path / "artifacts" / "runs").resolve()
    assert config.prompt_root == (tmp_path / "prompts" / "benchmark_v1").resolve()
    assert config.run_name == "demo-run"
    assert config.model_name == "demo-model"
    assert config.adapter == "mock"
    assert config.chain_manifest == (tmp_path / "artifacts" / "chain_pairs.jsonl").resolve()
    assert config.enable_oracle_track is True
    assert config.oracle_prompt_root == (tmp_path / "prompts" / "benchmark_oracle_v1").resolve()
    assert config.judge.backend == "openai"
    assert config.judge.prompt_root == (tmp_path / "prompts" / "judge_v1").resolve()
    assert config.judge.base_url == "http://judge.example/v1"
    assert config.judge.api_key_env == "CUSTOM_JUDGE_KEY"
    assert config.judge.model == "judge-model"
    assert config.judge.extra_body == {"thinking": {"type": "disabled"}}
    assert config.judge.invalid_json_retries == 3
    assert config.judge.concurrency == 2
    assert config.structurer.backend == "openai"
    assert config.structurer.prompt_root == (tmp_path / "prompts" / "structurer_v1").resolve()
    assert config.structurer.oracle_prompt_root == (tmp_path / "prompts" / "structurer_oracle_v1").resolve()
    assert config.structurer.base_url == "http://structurer.example/v1"
    assert config.structurer.api_key_env == "CUSTOM_STRUCTURER_KEY"
    assert config.structurer.model == "structurer-model"
    assert config.structurer.extra_body == {"thinking": {"type": "disabled"}}
    assert config.structurer.invalid_json_retries == 4
    assert config.structurer.concurrency == 5


def test_load_run_eval_config_requires_adapter(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
prompt_root = "prompts/benchmark_v1"

[judge]
prompt_root = "prompts/judge_v1"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[run_eval\]\.adapter is required"):
        load_run_eval_config(config_path)


def test_load_run_eval_config_requires_prompt_root(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
adapter = "mock"

[judge]
prompt_root = "prompts/judge_v1"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[run_eval\]\.prompt_root is required"):
        load_run_eval_config(config_path)


def test_load_run_eval_config_requires_oracle_prompt_roots_when_enabled(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
prompt_root = "prompts/benchmark_v1"
adapter = "mock"
enable_oracle_track = true

[judge]
prompt_root = "prompts/judge_v1"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[run_eval\]\.oracle_prompt_root is required"):
        load_run_eval_config(config_path)


def test_load_run_eval_config_requires_structurer_prompt_root(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
prompt_root = "prompts/benchmark_v1"
adapter = "mock"

[judge]
prompt_root = "prompts/judge_v1"

[structurer]
backend = "static-parse"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[structurer\]\.prompt_root is required"):
        load_run_eval_config(config_path)


def test_load_run_eval_config_requires_judge_prompt_root(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
prompt_root = "prompts/benchmark_v1"
adapter = "mock"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[judge\]\.prompt_root is required"):
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
                "supported_issue_count": 0,
                "ignored_unsupported_sample_count": 0,
            },
            {
                "num_prepared_samples": 2,
                "protocol_id": "expd_window_32s_2fps",
                "protocol_root": str(prepared_root / "expd_window_32s_2fps"),
                "supported_issue_count": 0,
                "ignored_unsupported_sample_count": 0,
            },
        ]

    monkeypatch.setattr("omnichain_eval.cli.build_prepared_data", fake_build_prepared_data)

    exit_code = main(["prepare-data", "--config", str(config_path)])

    assert exit_code == 0
    assert captured["data_root"] == (tmp_path / "data").resolve()
    assert captured["prepared_root"] == (tmp_path / "prepared_data").resolve()
    assert captured["protocols"] == ["main", "expd_window_32s_2fps"]
