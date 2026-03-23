"""Strict prompt template loading and rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    ALL_TASKS,
    TASK_AI_COACH,
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_MULTIPLE,
    TASK_SCOREBOARD_SINGLE,
    TASK_SCORE_PREDICTION,
    TASK_SPATIAL_IMAGINATION,
    TASK_STG,
    TASK_TEMPORAL_CAUSAL,
)
from .schema import ModelInput, PreparedSample, PromptMessage, RenderedPrompt

_SECTION_NAMES = {"system", "user"}
_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_ALLOWED_VARIABLES = {
    "question",
    "task_name",
    "task_level",
    "protocol_id",
    "num_sampled_frames",
    "sampled_index_range",
    "output_contract",
    "oracle_track_enabled",
}


class PromptTemplateError(ValueError):
    """Raised when a prompt pack or prompt template is invalid."""


@dataclass(slots=True)
class PromptTemplate:
    task_name: str
    path: Path
    system_template: str
    user_template: str


def _parse_template_sections(path: Path, text: str) -> tuple[str, str]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in text.splitlines():
        if line.startswith("# "):
            section_name = line[2:].strip()
            if section_name not in _SECTION_NAMES:
                raise PromptTemplateError(
                    f"{path}: unsupported top-level section {section_name!r}; "
                    "expected exactly '# system' and '# user'"
                )
            if section_name in sections:
                raise PromptTemplateError(f"{path}: duplicate section '# {section_name}'")
            current_section = section_name
            sections[current_section] = []
            continue
        if current_section is None:
            if line.strip():
                raise PromptTemplateError(
                    f"{path}: content appeared before the first top-level section"
                )
            continue
        sections[current_section].append(line)

    if set(sections) != _SECTION_NAMES:
        missing = sorted(_SECTION_NAMES - set(sections))
        raise PromptTemplateError(f"{path}: missing required section(s): {', '.join(missing)}")

    system_template = "\n".join(sections["system"]).strip()
    user_template = "\n".join(sections["user"]).strip()
    if not user_template:
        raise PromptTemplateError(f"{path}: '# user' section must not be empty")
    return system_template, user_template


def _validate_template_variables(path: Path, template_text: str) -> None:
    variables = set(_VARIABLE_PATTERN.findall(template_text))
    unknown = sorted(variables - _ALLOWED_VARIABLES)
    if unknown:
        raise PromptTemplateError(
            f"{path}: unsupported variable(s): {', '.join(unknown)}"
        )


def _load_prompt_template(task_name: str, path: Path) -> PromptTemplate:
    if not path.exists():
        raise PromptTemplateError(f"missing prompt template: {path}")
    text = path.read_text(encoding="utf-8")
    system_template, user_template = _parse_template_sections(path, text)
    _validate_template_variables(path, system_template)
    _validate_template_variables(path, user_template)
    return PromptTemplate(
        task_name=task_name,
        path=path,
        system_template=system_template,
        user_template=user_template,
    )


def load_prompt_pack(prompt_root: Path) -> dict[str, PromptTemplate]:
    if not prompt_root.exists():
        raise PromptTemplateError(f"prompt_root does not exist: {prompt_root}")
    if not prompt_root.is_dir():
        raise PromptTemplateError(f"prompt_root is not a directory: {prompt_root}")

    prompt_pack: dict[str, PromptTemplate] = {}
    for task_name in sorted(ALL_TASKS):
        prompt_pack[task_name] = _load_prompt_template(
            task_name,
            prompt_root / f"{task_name}.md",
        )
    return prompt_pack


def _output_contract(task_name: str) -> str:
    if task_name in {
        TASK_SCOREBOARD_MULTIPLE,
        TASK_SPATIAL_IMAGINATION,
        TASK_TEMPORAL_CAUSAL,
        TASK_SCORE_PREDICTION,
        TASK_AI_COACH,
    }:
        return 'Return JSON only: {"text": "..."}'
    if task_name == TASK_SCOREBOARD_SINGLE:
        return 'Return JSON only: {"text": "...", "bbox": [xtl, ytl, xbr, ybr]}'
    if task_name == TASK_OBJECTS_SPATIAL:
        return (
            'Return JSON only: {"text": "...", "bbox_a": [xtl, ytl, xbr, ybr], '
            '"bbox_b": [xtl, ytl, xbr, ybr]}'
        )
    if task_name in {TASK_CONTINUOUS_EVENTS, TASK_COMMENTARY}:
        return (
            'Return JSON only: {"segments": [{"start_sampled": 0, "end_sampled": 3, '
            '"text": "..."}]}'
        )
    if task_name == TASK_CONTINUOUS_ACTIONS:
        return (
            'Return JSON only: {"segments": [{"start_sampled": 0, "end_sampled": 3, '
            '"text": "..."}], "tracking": [{"frame_sampled": 0, "bbox_mot": [left, top, '
            'width, height]}]}'
        )
    if task_name == TASK_STG:
        return (
            'Return JSON only: {"time_window_sampled": [0, 4], "tracking": '
            '[{"frame_sampled": 0, "bbox_mot": [left, top, width, height]}]}'
        )
    raise PromptTemplateError(f"unsupported task_name for output contract: {task_name}")


def _sampled_index_range(sample: PreparedSample) -> str:
    if not sample.sampled_frames_original:
        raise PromptTemplateError(f"{sample.sample_id}: no sampled frames available")
    return f"0..{len(sample.sampled_frames_original) - 1}"


def _render_template_text(template_text: str, variables: dict[str, object], path: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in variables:
            raise PromptTemplateError(f"{path}: missing render variable {variable_name!r}")
        value = variables[variable_name]
        return str(value)

    return _VARIABLE_PATTERN.sub(replace, template_text).strip()


def render_prompt(
    prompt_pack: dict[str, PromptTemplate],
    sample: PreparedSample,
    *,
    oracle_track: bool = False,
) -> RenderedPrompt:
    try:
        template = prompt_pack[sample.task_name]
    except KeyError as exc:
        raise PromptTemplateError(f"missing prompt template for task {sample.task_name}") from exc

    variables: dict[str, object] = {
        "question": sample.question_text,
        "task_name": sample.task_name,
        "task_level": sample.task_level,
        "protocol_id": sample.protocol_id,
        "num_sampled_frames": len(sample.sampled_frames_original),
        "sampled_index_range": _sampled_index_range(sample),
        "output_contract": _output_contract(sample.task_name),
        "oracle_track_enabled": str(oracle_track).lower(),
    }
    system_prompt = _render_template_text(template.system_template, variables, template.path)
    user_prompt = _render_template_text(template.user_template, variables, template.path)
    return RenderedPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variables=variables,
    )


def build_model_input(
    sample: PreparedSample,
    rendered_prompt: RenderedPrompt,
    *,
    oracle_track: bool = False,
    conversation_history: list[PromptMessage] | None = None,
) -> ModelInput:
    messages: list[PromptMessage] = []
    if rendered_prompt.system_prompt:
        messages.append(PromptMessage(role="system", content=rendered_prompt.system_prompt))
    if conversation_history:
        messages.extend(conversation_history)
    messages.append(PromptMessage(role="user", content=rendered_prompt.user_prompt))
    return ModelInput(
        sample=sample,
        task_name=sample.task_name,
        frame_files=list(sample.frame_files),
        messages=messages,
        oracle_track=oracle_track,
    )


def serialize_model_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, (dict, list)):
        return json.dumps(raw_output, ensure_ascii=False)
    return str(raw_output)


def build_chain_history(
    upstream_sample: PreparedSample,
    upstream_raw_output: Any,
) -> list[PromptMessage]:
    return [
        PromptMessage(role="user", content=upstream_sample.question_text),
        PromptMessage(role="assistant", content=serialize_model_output(upstream_raw_output)),
    ]
