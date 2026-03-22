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
from .utils import ensure_directory, extract_json_object, stable_hash, write_json


class JudgeClient(ABC):
    """Base interface for judge backends."""

    @abstractmethod
    def judge(
        self,
        task_name: str,
        prompt_text: str,
        reference_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> JudgeDecision:
        raise NotImplementedError


class StaticJudgeClient(JudgeClient):
    """Deterministic test backend."""

    def __init__(self, *, always_pass: bool = True) -> None:
        self.always_pass = always_pass

    def judge(
        self,
        task_name: str,
        prompt_text: str,
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
    prompt_text: str,
    reference_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
) -> str:
    payload = {
        "task_name": task_name,
        "question_or_query": prompt_text,
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
    """OpenAI-compatible JSON judge client with on-disk caching."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = JUDGE_MODEL_DEFAULT,
        cache_dir: Path | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.cache_dir = ensure_directory(cache_dir) if cache_dir else None

    @classmethod
    def from_env(cls, *, cache_dir: Path | None = None) -> "OpenAIJudgeClient":
        base_url = os.environ.get("EVAL_JUDGE_BASE_URL")
        api_key = os.environ.get("EVAL_JUDGE_API_KEY")
        model = os.environ.get("EVAL_JUDGE_MODEL", JUDGE_MODEL_DEFAULT)
        if not base_url or not api_key:
            raise ValueError(
                "judge configuration is missing; set EVAL_JUDGE_BASE_URL and EVAL_JUDGE_API_KEY"
            )
        return cls(base_url=base_url, api_key=api_key, model=model, cache_dir=cache_dir)

    def _cache_path(self, request_payload: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{stable_hash(request_payload)}.json"

    def _parse_decision(self, raw_response: str) -> JudgeDecision:
        try:
            payload = extract_json_object(raw_response)
        except ValueError:
            return default_judge_fail("judge response was not valid JSON", raw_response)
        if not isinstance(payload, dict) or not JUDGE_JSON_KEYS.issubset(payload.keys()):
            return default_judge_fail("judge response did not match schema", raw_response)
        try:
            decision = JudgeDecision(
                correctness=int(payload["correctness"]),
                completeness=int(payload["completeness"]),
                faithfulness=int(payload["faithfulness"]),
                final_pass=int(payload["final_pass"]),
                confidence=str(payload["confidence"]),
                brief_reason=str(payload["brief_reason"]),
                raw_response=raw_response,
            )
        except Exception:  # noqa: BLE001
            return default_judge_fail("judge response could not be coerced to schema", raw_response)
        return decision

    def judge(
        self,
        task_name: str,
        prompt_text: str,
        reference_payload: dict[str, Any],
        prediction_payload: dict[str, Any],
    ) -> JudgeDecision:
        request_payload = {
            "model": self.model,
            "task_name": task_name,
            "prompt_text": prompt_text,
            "reference_payload": reference_payload,
            "prediction_payload": prediction_payload,
        }
        cache_path = self._cache_path(request_payload)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return self._parse_decision(cached["raw_response"])

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        task_name,
                        prompt_text,
                        reference_payload,
                        prediction_payload,
                    ),
                },
            ],
            temperature=0,
            top_p=1.0,
            max_tokens=256,
            n=1,
            seed=42,
            extra_body={"top_k": 1},
        )
        message = completion.choices[0].message.content
        if isinstance(message, list):
            raw_response = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in message
            )
        else:
            raw_response = message or ""
        if cache_path:
            write_json(cache_path, {"request": request_payload, "raw_response": raw_response})
        return self._parse_decision(raw_response)
