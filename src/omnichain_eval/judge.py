"""LLM-as-a-judge integration."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from openai import OpenAI

from .constants import (
    JUDGE_JSON_KEYS,
    JUDGE_MODEL_DEFAULT,
    JUDGE_REQUIRED_TASKS,
)
from .schema import JudgeDecision
from .template_pack import (
    TaskTemplate,
    TemplatePackError,
    load_task_template_pack,
    render_template_text,
)
from .utils import extract_json_object

_ALLOWED_JUDGE_VARIABLES = {
    "question_text",
    "reference_payload_json",
    "prediction_payload_json",
}

JudgePromptTemplateError = TemplatePackError
JudgePromptTemplate = TaskTemplate
JudgePromptPack = dict[str, TaskTemplate]


class JudgeClient(ABC):
    """Base interface for judge backends."""

    @abstractmethod
    def judge(
        self,
        task_name: str,
        question_text: str,
        reference_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> JudgeDecision:
        raise NotImplementedError


class JudgeResponseFormatError(ValueError):
    """Raised when the judge response format is not usable."""

    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class JudgeResponseFormatExhaustedError(RuntimeError):
    """Raised when the judge response remains unusable after retries."""

    def __init__(self, reason: str, raw_response: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_response = raw_response


class StaticJudgeClient(JudgeClient):
    """Deterministic test backend."""

    def __init__(self, *, always_pass: bool = True) -> None:
        self.always_pass = always_pass

    def judge(
        self,
        task_name: str,
        question_text: str,
        reference_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> JudgeDecision:
        final_pass = 1 if self.always_pass else 0
        return JudgeDecision(
            correctness=final_pass,
            completeness=final_pass,
            faithfulness=final_pass,
            final_pass=final_pass,
            confidence="high" if final_pass else "low",
            brief_reason="Static judge decision.",
            raw_response=None,
        )


def default_judge_fail(reason: str, raw_response: str | None = None) -> JudgeDecision:
    return JudgeDecision(
        correctness=0,
        completeness=0,
        faithfulness=0,
        final_pass=0,
        confidence="low",
        brief_reason=reason,
        raw_response=raw_response,
    )


def load_judge_prompt_pack(prompt_root: Path) -> JudgePromptPack:
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_JUDGE_VARIABLES,
        task_names=sorted(JUDGE_REQUIRED_TASKS),
    )


def _judge_variables(
    question_text: str,
    reference_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
) -> dict[str, str]:
    payload = {
        "question_text": question_text,
        "reference_payload_json": json.dumps(reference_payload, ensure_ascii=False, indent=2),
        "prediction_payload_json": json.dumps(prediction_payload, ensure_ascii=False, indent=2),
    }
    return payload


def render_judge_prompt(
    prompt_pack: JudgePromptPack,
    *,
    task_name: str,
    question_text: str,
    reference_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
) -> str:
    try:
        template = prompt_pack[task_name]
    except KeyError as exc:
        raise JudgePromptTemplateError(f"missing judge prompt template for task {task_name}") from exc
    variables = _judge_variables(
        question_text,
        reference_payload,
        prediction_payload,
    )
    return render_template_text(template.prompt_template, variables, template.path)


class OpenAIJudgeClient(JudgeClient):
    """OpenAI-compatible JSON judge client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        prompt_root: Path,
        model: str = JUDGE_MODEL_DEFAULT,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        invalid_json_retries: int = 0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.prompt_pack = load_judge_prompt_pack(prompt_root)
        self.model = model
        self.temperature = float(temperature)
        self.extra_body = {
            "enable_thinking": False,
            **dict(extra_body or {}),
        }
        self.invalid_json_retries = invalid_json_retries

    @classmethod
    def from_env(
        cls,
        *,
        prompt_root: Path,
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        invalid_json_retries: int = 0,
    ) -> "OpenAIJudgeClient":
        base_url = os.environ.get("EVAL_JUDGE_BASE_URL")
        api_key = os.environ.get("EVAL_JUDGE_API_KEY")
        model = os.environ.get("EVAL_JUDGE_MODEL", JUDGE_MODEL_DEFAULT)
        if not base_url or not api_key:
            raise ValueError(
                "judge configuration is missing; set EVAL_JUDGE_BASE_URL and EVAL_JUDGE_API_KEY"
            )
        return cls(
            base_url=base_url,
            api_key=api_key,
            prompt_root=prompt_root,
            model=model,
            temperature=temperature,
            extra_body=extra_body,
            invalid_json_retries=invalid_json_retries,
        )

    def _extract_payload(self, raw_response: str) -> dict[str, Any]:
        try:
            payload = extract_json_object(raw_response)
        except ValueError as exc:
            raise JudgeResponseFormatError("judge response was not valid JSON", raw_response) from exc
        if not isinstance(payload, dict):
            raise JudgeResponseFormatError("judge response was not a JSON object", raw_response)
        return payload

    def _parse_decision_from_payload(
        self,
        raw_response: str,
        payload: dict[str, Any],
    ) -> JudgeDecision:
        if not JUDGE_JSON_KEYS.issubset(payload.keys()):
            raise JudgeResponseFormatError("judge response did not match schema", raw_response)
        try:
            correctness = int(payload["correctness"])
            completeness = int(payload["completeness"])
            faithfulness = int(payload["faithfulness"])
            final_pass = int(payload["final_pass"])
            confidence = str(payload["confidence"]).strip()
            brief_reason = str(payload["brief_reason"]).strip()
        except Exception as exc:  # noqa: BLE001
            raise JudgeResponseFormatError(
                "judge response could not be coerced to schema",
                raw_response,
            ) from exc
        if {correctness, completeness, faithfulness, final_pass} - {0, 1}:
            raise JudgeResponseFormatError(
                "judge response contained non-binary scoring fields",
                raw_response,
            )
        if not confidence:
            raise JudgeResponseFormatError("judge response confidence was empty", raw_response)
        if not brief_reason:
            raise JudgeResponseFormatError("judge response brief_reason was empty", raw_response)
        return JudgeDecision(
            correctness=correctness,
            completeness=completeness,
            faithfulness=faithfulness,
            final_pass=final_pass,
            confidence=confidence,
            brief_reason=brief_reason,
            raw_response=raw_response,
        )

    def _parse_decision(self, raw_response: str) -> JudgeDecision:
        payload = self._extract_payload(raw_response)
        return self._parse_decision_from_payload(raw_response, payload)

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

    def judge(
        self,
        task_name: str,
        question_text: str,
        reference_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> JudgeDecision:
        prompt_text = render_judge_prompt(
            self.prompt_pack,
            task_name=task_name,
            question_text=question_text,
            reference_payload=reference_payload,
            prediction_payload=prediction_payload,
        )
        max_attempts = self.invalid_json_retries + 1
        last_raw_response: str | None = None
        last_error_reason = "judge response format was invalid"
        for attempt_index in range(max_attempts):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                extra_body=self.extra_body or None,
            )
            for raw_response in self._response_texts(completion):
                last_raw_response = raw_response
                try:
                    decision = self._parse_decision(raw_response)
                except JudgeResponseFormatError as exc:
                    last_error_reason = exc.reason
                    continue
                return decision
            if attempt_index + 1 >= max_attempts:
                raise JudgeResponseFormatExhaustedError(
                    f"{last_error_reason} after {max_attempts} attempt(s)",
                    last_raw_response,
                )
        raise JudgeResponseFormatExhaustedError(last_error_reason, last_raw_response)
