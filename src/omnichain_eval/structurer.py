"""LLM-based structured extraction for model outputs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .constants import (
    STRUCTURER_MODEL_DEFAULT,
    STRUCTURER_SYSTEM_PROMPT,
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
    "task_name",
    "question",
    "raw_output",
    "num_sampled_frames",
    "sampled_index_range",
    "output_schema",
}

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
    system_prompt: str
    user_prompt: str
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
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        max_tokens: int = 512,
        n: int = 1,
        seed: int = 42,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.n = n
        self.seed = seed

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
                {"role": "system", "content": rendered_prompt.system_prompt},
                {"role": "user", "content": rendered_prompt.user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.n,
            seed=self.seed,
            extra_body={"top_k": self.top_k},
        )
        return self._response_texts(completion)


def load_structurer_prompt_pack(prompt_root) -> dict[str, StructurerPromptTemplate]:
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_STRUCTURER_VARIABLES,
    )


def _sampled_index_range(sample: PreparedSample) -> str:
    if not sample.sampled_frames_original:
        raise StructurerPromptTemplateError(f"{sample.sample_id}: no sampled frames available")
    return f"0..{len(sample.sampled_frames_original) - 1}"


def _output_schema(task_name: str) -> str:
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
        return json.dumps({"text": "", "bbox_a": [], "bbox_b": []}, ensure_ascii=False, indent=2)
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
) -> RenderedStructurerPrompt:
    try:
        template = prompt_pack[sample.task_name]
    except KeyError as exc:
        raise StructurerPromptTemplateError(
            f"missing structurer prompt template for task {sample.task_name}"
        ) from exc
    variables: dict[str, Any] = {
        "task_name": sample.task_name,
        "question": sample.question_text,
        "raw_output": raw_output,
        "num_sampled_frames": len(sample.sampled_frames_original),
        "sampled_index_range": _sampled_index_range(sample),
        "output_schema": _output_schema(sample.task_name),
    }
    system_prompt = render_template_text(template.system_template, variables, template.path)
    user_prompt = render_template_text(template.user_template, variables, template.path)
    if not system_prompt:
        system_prompt = STRUCTURER_SYSTEM_PROMPT
    return RenderedStructurerPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        variables=variables,
    )


@dataclass(slots=True)
class StructurerService:
    backend: StructurerBackend
    prompt_pack: dict[str, StructurerPromptTemplate]
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
    ) -> StructuredPredictionResult:
        rendered_prompt = render_structurer_prompt(self.prompt_pack, sample, raw_output)
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
