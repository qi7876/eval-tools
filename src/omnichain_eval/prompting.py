"""Strict model-input prompt loading and rendering."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
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
from .schema import ModelInput, PreparedSample, PromptMessage, RenderedPrompt
from .template_pack import TaskTemplate, TemplatePackError, load_task_template_pack, render_template_text

_ALLOWED_INFERENCE_VARIABLES = {
    "question",
    "task_name",
    "task_level",
    "protocol_id",
    "num_sampled_frames",
    "sampled_index_range",
    "output_contract",
}

PromptTemplateError = TemplatePackError
PromptTemplate = TaskTemplate


def load_prompt_pack(prompt_root):
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_INFERENCE_VARIABLES,
    )


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
    if task_name == TASK_CONTINUOUS_EVENTS:
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


def render_prompt(
    prompt_pack: dict[str, PromptTemplate],
    sample: PreparedSample,
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
    }
    prompt_text = render_template_text(template.prompt_template, variables, template.path)
    return RenderedPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        prompt_text=prompt_text,
        variables=variables,
    )


def render_oracle_upstream_prompt(
    prompt_pack: dict[str, PromptTemplate],
    sample: PreparedSample,
) -> RenderedPrompt:
    if sample.task_name not in {TASK_CONTINUOUS_ACTIONS, TASK_STG}:
        raise PromptTemplateError(
            f"oracle upstream prompt is only supported for {TASK_CONTINUOUS_ACTIONS} and {TASK_STG}"
        )
    rendered = render_prompt(prompt_pack, sample)
    tracking_gt = sample.reference_payload.get("tracking_gt_sampled")
    if not isinstance(tracking_gt, list):
        raise PromptTemplateError(f"{sample.sample_id}: missing tracking_gt_sampled for OracleTrack")
    note = (
        "Use the provided GT tracking as reference when answering the current upstream question."
    )
    if sample.task_name == TASK_STG:
        note += " Predict `time_window_sampled` yourself; only tracking is oracle."
    oracle_block = (
        "OracleTrack rerun.\n"
        f"{note}\n"
        "Here is the GT tracking in sampled-frame coordinates:\n"
        f"```json\n{json.dumps(tracking_gt, ensure_ascii=False, indent=2)}\n```"
    )
    return RenderedPrompt(
        task_name=rendered.task_name,
        template_path=rendered.template_path,
        prompt_text=f"{rendered.prompt_text}\n\n{oracle_block}",
        variables=rendered.variables,
    )


def build_model_input(
    sample: PreparedSample,
    rendered_prompt: RenderedPrompt,
    *,
    conversation_history: list[PromptMessage] | None = None,
) -> ModelInput:
    messages: list[PromptMessage] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append(PromptMessage(role="user", content=rendered_prompt.prompt_text))
    return ModelInput(
        sample=sample,
        messages=messages,
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
