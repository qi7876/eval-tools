"""Strict task-template loading and rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import ALL_TASKS

_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class TemplatePackError(ValueError):
    """Raised when a task template pack is invalid."""


@dataclass(slots=True)
class TaskTemplate:
    task_name: str
    path: Path
    prompt_template: str


def load_markdown_prompt_template(
    path: Path,
    *,
    allowed_variables: set[str],
) -> TaskTemplate:
    if not path.exists():
        raise TemplatePackError(f"prompt template does not exist: {path}")
    if not path.is_file():
        raise TemplatePackError(f"prompt template is not a file: {path}")
    text = path.read_text(encoding="utf-8")
    prompt_template = text.strip()
    if not prompt_template:
        raise TemplatePackError(f"{path}: prompt template must not be empty")
    _validate_template_variables(path, prompt_template, allowed_variables)
    return TaskTemplate(
        task_name=path.stem,
        path=path,
        prompt_template=prompt_template,
    )


def _validate_template_variables(path: Path, template_text: str, allowed_variables: set[str]) -> None:
    variables = set(_VARIABLE_PATTERN.findall(template_text))
    unknown = sorted(variables - allowed_variables)
    if unknown:
        raise TemplatePackError(
            f"{path}: unsupported variable(s): {', '.join(unknown)}"
        )


def load_task_template_pack(
    prompt_root: Path,
    *,
    allowed_variables: set[str],
) -> dict[str, TaskTemplate]:
    if not prompt_root.exists():
        raise TemplatePackError(f"prompt_root does not exist: {prompt_root}")
    if not prompt_root.is_dir():
        raise TemplatePackError(f"prompt_root is not a directory: {prompt_root}")

    prompt_pack: dict[str, TaskTemplate] = {}
    for task_name in sorted(ALL_TASKS):
        path = prompt_root / f"{task_name}.md"
        if not path.exists():
            raise TemplatePackError(f"missing prompt template: {path}")
        prompt_pack[task_name] = load_markdown_prompt_template(
            path,
            allowed_variables=allowed_variables,
        )
    return prompt_pack


def render_template_text(template_text: str, variables: dict[str, Any], path: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in variables:
            raise TemplatePackError(f"{path}: missing render variable {variable_name!r}")
        return str(variables[variable_name])

    return _VARIABLE_PATTERN.sub(replace, template_text).strip()
