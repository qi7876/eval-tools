"""TOML-backed command configuration."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import JUDGE_MODEL_DEFAULT, STRUCTURER_MODEL_DEFAULT


def _load_toml(path: Path) -> tuple[dict[str, Any], Path]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return payload, path.parent


def _table(payload: dict[str, Any], name: str) -> dict[str, Any]:
    raw = payload.get(name, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"TOML section [{name}] must be a table")
    return raw


def _resolve_path(base_dir: Path, raw: str | None, *, default: str | None = None) -> Path | None:
    value = raw if raw is not None else default
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_path_list(base_dir: Path, raw: list[str] | None, *, default: list[str] | None = None) -> list[Path]:
    values = raw if raw is not None else default or []
    return [_resolve_path(base_dir, value) for value in values if value is not None]  # type: ignore[list-item]


@dataclass(slots=True)
class ValidateDataConfig:
    data_root: Path = Path("data")


@dataclass(slots=True)
class BuildChainManifestConfig:
    data_root: Path = Path("data")
    out: Path = Path("artifacts/chain_pairs.jsonl")


@dataclass(slots=True)
class PrepareDataConfig:
    data_root: Path = Path("data")
    prepared_root: Path = Path("prepared_data")
    protocols: list[str] = field(default_factory=lambda: ["main"])


@dataclass(slots=True)
class JudgeConfig:
    backend: str = "openai"
    prompt_path: Path | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "EVAL_JUDGE_API_KEY"
    model: str = JUDGE_MODEL_DEFAULT
    extra_body: dict[str, Any] = field(default_factory=dict)
    invalid_json_retries: int = 0
    concurrency: int = 1

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return os.environ.get("EVAL_JUDGE_BASE_URL")

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


@dataclass(slots=True)
class StructurerConfig:
    backend: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "EVAL_STRUCTURER_API_KEY"
    model: str = STRUCTURER_MODEL_DEFAULT
    extra_body: dict[str, Any] = field(default_factory=dict)
    invalid_json_retries: int = 0
    concurrency: int = 1
    prompt_root: Path | None = None

    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        return os.environ.get("EVAL_STRUCTURER_BASE_URL")

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


@dataclass(slots=True)
class RunEvalConfig:
    prepared_root: Path = Path("prepared_data")
    protocol: str = "main"
    artifacts_root: Path = Path("artifacts/runs")
    prompt_root: Path | None = None
    run_name: str | None = None
    model_name: str | None = None
    chain_manifest: Path | None = None
    enable_oracle_track: bool = False
    adapter: str | None = None
    structurer: StructurerConfig = field(default_factory=StructurerConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def load_validate_data_config(path: Path) -> ValidateDataConfig:
    payload, base_dir = _load_toml(path)
    table = _table(payload, "validate_data")
    return ValidateDataConfig(
        data_root=_resolve_path(base_dir, table.get("data_root"), default="data"),  # type: ignore[arg-type]
    )


def load_build_chain_manifest_config(path: Path) -> BuildChainManifestConfig:
    payload, base_dir = _load_toml(path)
    table = _table(payload, "build_chain_manifest")
    return BuildChainManifestConfig(
        data_root=_resolve_path(base_dir, table.get("data_root"), default="data"),  # type: ignore[arg-type]
        out=_resolve_path(
            base_dir,
            table.get("out"),
            default="artifacts/chain_pairs.jsonl",
        ),  # type: ignore[arg-type]
    )


def load_prepare_data_config(path: Path) -> PrepareDataConfig:
    payload, base_dir = _load_toml(path)
    table = _table(payload, "prepare_data")
    protocols = table.get("protocols", ["main"])
    if not isinstance(protocols, list) or not all(isinstance(value, str) for value in protocols):
        raise ValueError("[prepare_data].protocols must be a list of strings")
    return PrepareDataConfig(
        data_root=_resolve_path(base_dir, table.get("data_root"), default="data"),  # type: ignore[arg-type]
        prepared_root=_resolve_path(
            base_dir,
            table.get("prepared_root"),
            default="prepared_data",
        ),  # type: ignore[arg-type]
        protocols=list(protocols),
    )


def _load_judge_config(base_dir: Path, payload: dict[str, Any]) -> JudgeConfig:
    table = _table(payload, "judge")
    extra_body = table.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("[judge].extra_body must be a table")
    config = JudgeConfig(
        backend=str(table.get("backend", "openai")),
        prompt_path=_resolve_path(base_dir, table.get("prompt_path")),
        base_url=table.get("base_url"),
        api_key=table.get("api_key"),
        api_key_env=str(table.get("api_key_env", "EVAL_JUDGE_API_KEY")),
        model=str(table.get("model", JUDGE_MODEL_DEFAULT)),
        extra_body=extra_body,
        invalid_json_retries=int(table.get("invalid_json_retries", 0)),
        concurrency=int(table.get("concurrency", 1)),
    )
    if config.backend not in {"openai", "static-pass", "static-fail"}:
        raise ValueError("[judge].backend must be one of: openai, static-pass, static-fail")
    if config.invalid_json_retries < 0:
        raise ValueError("[judge].invalid_json_retries must be >= 0")
    if config.concurrency < 1:
        raise ValueError("[judge].concurrency must be >= 1")
    if config.prompt_path is None:
        raise ValueError("[judge].prompt_path is required")
    return config


def _load_structurer_config(base_dir: Path, payload: dict[str, Any]) -> StructurerConfig:
    table = _table(payload, "structurer")
    extra_body = table.get("extra_body", {})
    if not isinstance(extra_body, dict):
        raise ValueError("[structurer].extra_body must be a table")
    config = StructurerConfig(
        backend=str(table.get("backend", "openai")),
        base_url=table.get("base_url"),
        api_key=table.get("api_key"),
        api_key_env=str(table.get("api_key_env", "EVAL_STRUCTURER_API_KEY")),
        model=str(table.get("model", STRUCTURER_MODEL_DEFAULT)),
        extra_body=extra_body,
        invalid_json_retries=int(table.get("invalid_json_retries", 0)),
        concurrency=int(table.get("concurrency", 1)),
        prompt_root=_resolve_path(base_dir, table.get("prompt_root")),
    )
    if config.backend not in {"openai", "static-parse"}:
        raise ValueError("[structurer].backend must be one of: openai, static-parse")
    if config.invalid_json_retries < 0:
        raise ValueError("[structurer].invalid_json_retries must be >= 0")
    if config.concurrency < 1:
        raise ValueError("[structurer].concurrency must be >= 1")
    if config.prompt_root is None:
        raise ValueError("[structurer].prompt_root is required")
    return config


def load_run_eval_config(path: Path) -> RunEvalConfig:
    payload, base_dir = _load_toml(path)
    table = _table(payload, "run_eval")
    config = RunEvalConfig(
        prepared_root=_resolve_path(
            base_dir,
            table.get("prepared_root"),
            default="prepared_data",
        ),  # type: ignore[arg-type]
        protocol=str(table.get("protocol", "main")),
        artifacts_root=_resolve_path(
            base_dir,
            table.get("artifacts_root"),
            default="artifacts/runs",
        ),  # type: ignore[arg-type]
        prompt_root=_resolve_path(base_dir, table.get("prompt_root")),
        run_name=table.get("run_name"),
        model_name=table.get("model_name"),
        chain_manifest=_resolve_path(base_dir, table.get("chain_manifest")),
        enable_oracle_track=bool(table.get("enable_oracle_track", False)),
        adapter=table.get("adapter"),
        structurer=_load_structurer_config(base_dir, payload),
        judge=_load_judge_config(base_dir, payload),
    )
    if not config.adapter:
        raise ValueError("[run_eval].adapter is required")
    if config.prompt_root is None:
        raise ValueError("[run_eval].prompt_root is required")
    return config
