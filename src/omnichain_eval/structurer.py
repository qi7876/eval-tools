"""LLM-based structured extraction for model outputs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .constants import (
    STRUCTURER_MODEL_DEFAULT,
    TASK_AI_COACH,
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
from .normalize import validate_structured_prediction
from .schema import PreparedSample, StructuredPredictionResult
from .template_pack import TaskTemplate, TemplatePackError, load_task_template_pack, render_template_text
from .utils import extract_json_object

_ALLOWED_STRUCTURER_VARIABLES = {
    "raw_output",
    "output_schema",
    "required_object_labels_json",
}

_ORACLE_UPSTREAM_TASKS = [TASK_CONTINUOUS_ACTIONS, TASK_STG]

StructurerPromptTemplateError = TemplatePackError
StructurerPromptTemplate = TaskTemplate


class StructurerResponseFormatError(ValueError):
    """Raised when a structurer response is not usable."""

    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class StructurerResponseFormatExhaustedError(RuntimeError):
    """Raised when the structurer response remains unusable after retries."""

    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


@dataclass(slots=True)
class RenderedStructurerPrompt:
    task_name: str
    template_path: str
    prompt_text: str
    variables: dict[str, Any]


class StructurerBackend(ABC):
    """Low-level backend for structurer completions."""

    @abstractmethod
    def complete(
        self,
        *,
        sample: PreparedSample,
        raw_output: str,
        rendered_prompt: RenderedStructurerPrompt,
    ) -> list[str]:
        raise NotImplementedError


class StaticParseStructurerBackend(StructurerBackend):
    """Test backend that forwards the raw output into the parsing/validation stage."""

    def complete(
        self,
        *,
        sample: PreparedSample,
        raw_output: str,
        rendered_prompt: RenderedStructurerPrompt,
    ) -> list[str]:
        return [raw_output]


class OpenAIStructurerBackend(StructurerBackend):
    """OpenAI-compatible backend for structurer prompts."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = STRUCTURER_MODEL_DEFAULT,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.extra_body = dict(extra_body or {})

    def _response_texts(self, completion: Any) -> list[str]:
        responses: list[str] = []
        for choice in getattr(completion, "choices", []):
            message = choice.message.content
            if isinstance(message, list):
                responses.append(
                    "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in message
                    )
                )
            else:
                responses.append(message or "")
        return responses or [""]

    def complete(
        self,
        *,
        sample: PreparedSample,
        raw_output: str,
        rendered_prompt: RenderedStructurerPrompt,
    ) -> list[str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": rendered_prompt.prompt_text},
            ],
            extra_body=self.extra_body or None,
        )
        return self._response_texts(completion)


def load_structurer_prompt_pack(prompt_root) -> dict[str, StructurerPromptTemplate]:
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_STRUCTURER_VARIABLES,
    )


def load_oracle_structurer_prompt_pack(prompt_root) -> dict[str, StructurerPromptTemplate]:
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_STRUCTURER_VARIABLES,
        task_names=_ORACLE_UPSTREAM_TASKS,
    )


def _required_object_labels_json(sample: PreparedSample) -> str:
    if sample.task_name != TASK_OBJECTS_SPATIAL:
        return "[]"
    objects = sample.reference_payload.get("objects")
    if not isinstance(objects, list):
        raise StructurerPromptTemplateError(
            f"{sample.sample_id}: missing objects in reference_payload"
        )
    labels = [str(obj["label"]).strip() for obj in objects]
    if any(not label for label in labels):
        raise StructurerPromptTemplateError(f"{sample.sample_id}: object labels must be non-empty")
    return json.dumps(labels, ensure_ascii=False)


def _output_schema(task_name: str, *, oracle_upstream: bool = False) -> str:
    if oracle_upstream:
        if task_name == TASK_CONTINUOUS_ACTIONS:
            return json.dumps({"segments": []}, ensure_ascii=False, indent=2)
        if task_name == TASK_STG:
            return json.dumps({"time_window_sampled": []}, ensure_ascii=False, indent=2)
        raise StructurerPromptTemplateError(
            f"oracle upstream structurer schema is unsupported for {task_name}"
        )
    if task_name in {
        TASK_SCOREBOARD_MULTIPLE,
        TASK_SPATIAL_IMAGINATION,
        TASK_TEMPORAL_CAUSAL,
        TASK_SCORE_PREDICTION,
        TASK_AI_COACH,
    }:
        return json.dumps({"text": ""}, ensure_ascii=False, indent=2)
    if task_name == TASK_SCOREBOARD_SINGLE:
        return json.dumps({"text": "", "bbox": []}, ensure_ascii=False, indent=2)
    if task_name == TASK_OBJECTS_SPATIAL:
        return json.dumps(
            {"text": "", "objects": [{"label": "", "bbox": []}]},
            ensure_ascii=False,
            indent=2,
        )
    if task_name == TASK_CONTINUOUS_EVENTS:
        return json.dumps({"segments": []}, ensure_ascii=False, indent=2)
    if task_name == TASK_CONTINUOUS_ACTIONS:
        return json.dumps({"segments": [], "tracking": []}, ensure_ascii=False, indent=2)
    if task_name == TASK_STG:
        return json.dumps({"time_window_sampled": [], "tracking": []}, ensure_ascii=False, indent=2)
    raise StructurerPromptTemplateError(f"unsupported task_name for structurer schema: {task_name}")


def render_structurer_prompt(
    prompt_pack: dict[str, StructurerPromptTemplate],
    sample: PreparedSample,
    raw_output: str,
    *,
    oracle_upstream: bool = False,
) -> RenderedStructurerPrompt:
    try:
        template = prompt_pack[sample.task_name]
    except KeyError as exc:
        raise StructurerPromptTemplateError(
            f"missing structurer prompt template for task {sample.task_name}"
        ) from exc
    variables: dict[str, Any] = {
        "raw_output": raw_output,
        "output_schema": _output_schema(sample.task_name, oracle_upstream=oracle_upstream),
        "required_object_labels_json": _required_object_labels_json(sample),
    }
    prompt_text = render_template_text(template.prompt_template, variables, template.path)
    return RenderedStructurerPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        prompt_text=prompt_text,
        variables=variables,
    )


@dataclass(slots=True)
class StructurerService:
    backend: StructurerBackend
    prompt_pack: dict[str, StructurerPromptTemplate]
    oracle_prompt_pack: dict[str, StructurerPromptTemplate] | None = None
    invalid_json_retries: int = 0

    def _parse_payload(self, raw_response: str) -> dict[str, Any]:
        try:
            payload = extract_json_object(raw_response)
        except ValueError as exc:
            raise StructurerResponseFormatError(
                "structurer response was not valid JSON",
                raw_response,
            ) from exc
        if not isinstance(payload, dict):
            raise StructurerResponseFormatError(
                "structurer response was not a JSON object",
                raw_response,
            )
        return payload

    def structure(
        self,
        sample: PreparedSample,
        raw_output: str,
        *,
        oracle_upstream: bool = False,
    ) -> StructuredPredictionResult:
        prompt_pack = self.oracle_prompt_pack if oracle_upstream else self.prompt_pack
        if prompt_pack is None:
            raise StructurerPromptTemplateError("oracle_prompt_pack is required for OracleTrack structuring")
        rendered_prompt = render_structurer_prompt(
            prompt_pack,
            sample,
            raw_output,
            oracle_upstream=oracle_upstream,
        )
        max_attempts = self.invalid_json_retries + 1
        last_raw_response: str | None = None
        last_error_reason = "structurer response format was invalid"
        for attempt_index in range(max_attempts):
            for raw_response in self.backend.complete(
                sample=sample,
                raw_output=raw_output,
                rendered_prompt=rendered_prompt,
            ):
                last_raw_response = raw_response
                try:
                    payload = self._parse_payload(raw_response)
                except StructurerResponseFormatError as exc:
                    last_error_reason = exc.reason
                    continue
                validation = validate_structured_prediction(
                    sample,
                    raw_output,
                    payload,
                    structurer_raw_response=raw_response,
                    oracle_upstream=oracle_upstream,
                )
                if validation.errors:
                    last_error_reason = "; ".join(validation.errors)
                    continue
                return validation
            if attempt_index + 1 >= max_attempts:
                raise StructurerResponseFormatExhaustedError(
                    f"{last_error_reason} after {max_attempts} attempt(s)",
                    last_raw_response,
                )
        raise StructurerResponseFormatExhaustedError(last_error_reason, last_raw_response)
