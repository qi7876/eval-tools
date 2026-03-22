"""Shared helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(inner) for key, inner in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(to_jsonable(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty string cannot be parsed as JSON")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", stripped, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))
    start = min(
        (index for index in (stripped.find("{"), stripped.find("[")) if index != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError("no JSON object or array found")
    for end in range(len(stripped), start, -1):
        snippet = stripped[start:end]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("could not recover JSON payload")


def canonical_interval(interval: list[int] | tuple[int, int]) -> tuple[int, int]:
    start, end = int(interval[0]), int(interval[1])
    if start > end:
        start, end = end, start
    return start, end


def parse_interval_string(value: str) -> tuple[int, int]:
    pieces = value.split("-", maxsplit=1)
    if len(pieces) != 2:
        raise ValueError(f"invalid interval string: {value}")
    return canonical_interval((int(pieces[0]), int(pieces[1])))


def clip_index(value: int, total_frames: int) -> int:
    return min(max(int(value), 0), total_frames - 1)


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def sample_bundle_dir(samples_root: Path, video_key: str, annotation_id: str) -> Path:
    return samples_root / Path(video_key) / annotation_id

