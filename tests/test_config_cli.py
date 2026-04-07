from pathlib import Path
from textwrap import dedent

import pytest

from omnichain_eval.cli import main
from omnichain_eval.config import load_prepare_data_config, load_run_eval_config
from omnichain_eval.protocols import ALL_PROTOCOLS


def _write_protocol_module(tmp_path: Path) -> str:
    module_name = "custom_protocols_config"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        dedent(
            """
            from omnichain_eval.protocols import BaseProtocol


            class NativeEightProtocol(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Eight-frame custom protocol."

                def sample_frames(self, sample):
                    del sample
                    return [0, 1, 2, 3, 4, 5, 6, 7]

                def to_manifest_dict(self):
                    return {
                        **super().to_manifest_dict(),
                        "sampling": "uniform",
                        "frame_budget": 8,
                    }


            class NativeEightProtocolAlt(BaseProtocol):
                @property
                def protocol_id(self) -> str:
                    return "native_uniform_8"

                @property
                def description(self) -> str:
                    return "Alternate custom protocol."

                def sample_frames(self, sample):
                    del sample
                    return [0, 1, 2, 3]

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
temperature = 0
invalid_json_retries = 3
concurrency = 2

[judge.extra_body]
provider_hint = "judge"

[structurer]
backend = "openai"
prompt_root = "../prompts/structurer_v1"
oracle_prompt_root = "../prompts/structurer_oracle_v1"
base_url = "http://structurer.example/v1"
api_key_env = "CUSTOM_STRUCTURER_KEY"
model = "structurer-model"
temperature = 0
invalid_json_retries = 4
concurrency = 5

[structurer.extra_body]
provider_hint = "structurer"
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
    assert config.judge.temperature == 0.0
    assert config.judge.extra_body == {
        "enable_thinking": False,
        "provider_hint": "judge",
    }
    assert config.judge.invalid_json_retries == 3
    assert config.judge.concurrency == 2
    assert config.structurer.backend == "openai"
    assert config.structurer.prompt_root == (tmp_path / "prompts" / "structurer_v1").resolve()
    assert config.structurer.oracle_prompt_root == (tmp_path / "prompts" / "structurer_oracle_v1").resolve()
    assert config.structurer.base_url == "http://structurer.example/v1"
    assert config.structurer.api_key_env == "CUSTOM_STRUCTURER_KEY"
    assert config.structurer.model == "structurer-model"
    assert config.structurer.temperature == 0.0
    assert config.structurer.extra_body == {
        "enable_thinking": False,
        "provider_hint": "structurer",
    }
    assert config.structurer.invalid_json_retries == 4
    assert config.structurer.concurrency == 5


def test_load_run_eval_config_uses_qwen_defaults_for_openai_sections(tmp_path):
    config_path = tmp_path / "defaults.toml"
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
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    config = load_run_eval_config(config_path)

    assert config.judge.model == "qwen3.5-397b-a17b"
    assert config.judge.temperature == 0.0
    assert config.judge.extra_body == {"enable_thinking": False}
    assert config.structurer.model == "qwen3.5-397b-a17b"
    assert config.structurer.temperature == 0.0
    assert config.structurer.extra_body == {"enable_thinking": False}


def test_load_run_eval_config_rejects_non_numeric_temperature(tmp_path):
    config_path = tmp_path / "invalid_temperature.toml"
    config_path.write_text(
        """
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
prompt_root = "prompts/benchmark_v1"
adapter = "mock"

[judge]
prompt_root = "prompts/judge_v1"
temperature = "cold"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[judge\]\.temperature must be a number"):
        load_run_eval_config(config_path)


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
protocols = ["main", "main"]
media_formats = ["frames", "sampled_video"]
generate_oracle_visual_media = true
workers = 4
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_build_prepared_data(
        data_root: Path,
        prepared_root: Path,
        protocols: list[str],
        *,
        media_formats: list[str] | None,
        generate_oracle_visual_media: bool,
        workers: int,
    ):
        captured["data_root"] = data_root
        captured["prepared_root"] = prepared_root
        captured["protocols"] = protocols
        captured["media_formats"] = media_formats
        captured["generate_oracle_visual_media"] = generate_oracle_visual_media
        captured["workers"] = workers
        return [
            {
                "num_prepared_samples": 3,
                "protocol_id": "main",
                "protocol_root": str(prepared_root / "main"),
                "supported_issue_count": 0,
                "ignored_unsupported_sample_count": 0,
            },
        ]

    monkeypatch.setattr("omnichain_eval.cli.build_prepared_data", fake_build_prepared_data)

    exit_code = main(["prepare-data", "--config", str(config_path)])

    assert exit_code == 0
    assert captured["data_root"] == (tmp_path / "data").resolve()
    assert captured["prepared_root"] == (tmp_path / "prepared_data").resolve()
    assert captured["protocols"] == ["main"]
    assert captured["media_formats"] == ["frames", "sampled_video"]
    assert captured["generate_oracle_visual_media"] is True
    assert captured["workers"] == 4


def test_load_prepare_data_config_parses_workers(tmp_path):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main"]
media_formats = ["frames", "sampled_video", "frames"]
generate_oracle_visual_media = true
workers = 8
""".strip(),
        encoding="utf-8",
    )

    config = load_prepare_data_config(config_path)

    assert config.data_root == (tmp_path / "data").resolve()
    assert config.prepared_root == (tmp_path / "prepared_data").resolve()
    assert config.protocols == ["main"]
    assert config.media_formats == ["frames", "sampled_video"]
    assert config.generate_oracle_visual_media is True
    assert config.workers == 8


def test_load_prepare_data_config_rejects_invalid_media_formats(tmp_path):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main"]
media_formats = ["frames", "raw_video"]
workers = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\[prepare_data\]\.media_formats entries must be one of: frames, sampled_video",
    ):
        load_prepare_data_config(config_path)


def test_load_prepare_data_config_requires_dual_media_for_oracle_visual(tmp_path):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main"]
media_formats = ["frames"]
generate_oracle_visual_media = true
workers = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\[prepare_data\]\.generate_oracle_visual_media requires media_formats to include both frames and sampled_video",
    ):
        load_prepare_data_config(config_path)


def test_load_prepare_data_config_rejects_non_positive_workers(tmp_path):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        """
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main"]
workers = 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[prepare_data\]\.workers must be >= 1"):
        load_prepare_data_config(config_path)


def test_load_prepare_data_config_accepts_custom_protocol_spec(tmp_path, monkeypatch):
    module_name = _write_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        f"""
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main", "{module_name}:NativeEightProtocol"]
workers = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_prepare_data_config(config_path)

    assert config.protocols == ["main", f"{module_name}:NativeEightProtocol"]


def test_load_prepare_data_config_rejects_duplicate_protocol_ids(tmp_path, monkeypatch):
    module_name = _write_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        f"""
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["{module_name}:NativeEightProtocol", "{module_name}:NativeEightProtocolAlt"]
workers = 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"\[prepare_data\]\.protocols must resolve to unique protocol_id values",
    ):
        load_prepare_data_config(config_path)


def test_load_run_eval_config_accepts_custom_protocol_spec(tmp_path, monkeypatch):
    module_name = _write_protocol_module(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    config_path = tmp_path / "run_eval.toml"
    config_path.write_text(
        f"""
[run_eval]
prepared_root = "prepared_data"
protocol = "{module_name}:NativeEightProtocol"
prompt_root = "prompts/benchmark_v1"
adapter = "mock"

[judge]
prompt_root = "prompts/judge_v1"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
""".strip(),
        encoding="utf-8",
    )

    config = load_run_eval_config(config_path)

    assert config.protocol == f"{module_name}:NativeEightProtocol"


EXAMPLE_CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "examples"


def test_example_prepare_config_targets_main_only():
    main_config = load_prepare_data_config(EXAMPLE_CONFIG_ROOT / "prepare_main.toml")

    assert main_config.protocols == ["main"]
    assert main_config.media_formats == ["frames", "sampled_video"]
    assert main_config.generate_oracle_visual_media is True
    assert not (EXAMPLE_CONFIG_ROOT / "prepare_experiment_d.toml").exists()


def test_example_run_eval_configs_cover_all_protocols_with_dashscope_settings():
    for protocol_id in sorted(ALL_PROTOCOLS):
        config_path = EXAMPLE_CONFIG_ROOT / f"run_eval_{protocol_id}.toml"
        assert config_path.exists(), f"missing example config for protocol {protocol_id}"
        config = load_run_eval_config(config_path)

        assert config.protocol == protocol_id
        assert config.chain_manifest == (
            (EXAMPLE_CONFIG_ROOT / "../../artifacts/chain_pairs.jsonl").resolve()
        )
        assert config.judge.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert config.judge.api_key_env == "DASHSCOPE_API_KEY"
        assert config.structurer.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert config.structurer.api_key_env == "DASHSCOPE_API_KEY"
