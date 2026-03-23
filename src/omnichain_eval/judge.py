"""LLM-as-a-judge integration."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from .constants import (
    JUDGE_JSON_KEYS,
    JUDGE_MODEL_DEFAULT,
    JUDGE_SYSTEM_PROMPT,
    TASK_AI_COACH,
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_MULTIPLE,
    TASK_SCOREBOARD_SINGLE,
    TASK_SCORE_PREDICTION,
    TASK_SPATIAL_IMAGINATION,
    TASK_TEMPORAL_CAUSAL,
)
from .schema import JudgeDecision
from .utils import extract_json_object


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


def _task_instruction(task_name: str) -> str:
    if task_name in {
        TASK_SCOREBOARD_SINGLE,
        TASK_SCOREBOARD_MULTIPLE,
        TASK_OBJECTS_SPATIAL,
        TASK_SPATIAL_IMAGINATION,
        TASK_SCORE_PREDICTION,
    }:
        return (
            "Rule: all core slots must be correct. Key entity, direction, score, rank, "
            "or relation errors cause failure."
        )
    if task_name == TASK_TEMPORAL_CAUSAL:
        return (
            "Rule: pass if the main cause is correct and there is no key hallucination. "
            "The main cause must not be replaced by a side effect."
        )
    if task_name in {TASK_CONTINUOUS_ACTIONS, TASK_CONTINUOUS_EVENTS, TASK_COMMENTARY}:
        return (
            "Rule: judge time alignment and text jointly. Key actions or events must be "
            "covered, temporal order must be correct, and rough alignment must be reasonable."
        )
    if task_name == TASK_AI_COACH:
        return (
            "Rule: multiple valid answers may exist. Pass if the suggestion is relevant, "
            "actionable, and aligned with the reference intent."
        )
    return "Apply the benchmark rubric strictly."


def _build_user_prompt(
    task_name: str,
    question_text: str,
    reference_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
) -> str:
    payload = {
        "task_name": task_name,
        "question_or_query": question_text,
        "task_specific_rule": _task_instruction(task_name),
        "reference": reference_payload,
        "prediction": prediction_payload,
        "required_json_schema": {
            "correctness": 1,
            "completeness": 1,
            "faithfulness": 1,
            "final_pass": 1,
            "confidence": "high",
            "brief_reason": "Prediction matches the reference answer with acceptable paraphrasing.",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


class OpenAIJudgeClient(JudgeClient):
    """OpenAI-compatible JSON judge client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = JUDGE_MODEL_DEFAULT,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        max_tokens: int = 256,
        n: int = 1,
        seed: int = 42,
        invalid_json_retries: int = 0,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.n = n
        self.seed = seed
        self.invalid_json_retries = invalid_json_retries

    @classmethod
    def from_env(
        cls,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        max_tokens: int = 256,
        n: int = 1,
        seed: int = 42,
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
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
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
        max_attempts = self.invalid_json_retries + 1
        last_raw_response: str | None = None
        last_error_reason = "judge response format was invalid"
        for attempt_index in range(max_attempts):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _build_user_prompt(
                            task_name,
                            question_text,
                            reference_payload,
                            prediction_payload,
                        ),
                    },
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=self.n,
                seed=self.seed,
                extra_body={"top_k": self.top_k},
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
