"""Strict model-input prompt loading and rendering."""

from __future__ import annotations

import json
from typing import Any

from .constants import (
    ORACLE_EXPERIMENT_B_VARIANTS,
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
    "num_sampled_frames",
    "sampled_index_range",
    "sampled_video_fps",
    "output_contract",
    "required_object_labels_json",
}

_ALLOWED_ORACLE_INFERENCE_VARIABLES = _ALLOWED_INFERENCE_VARIABLES | {
    "oracle_tracking_explanation",
    "oracle_tracking_json",
    "oracle_visual_explanation",
}

_ORACLE_UPSTREAM_TASKS = [TASK_CONTINUOUS_ACTIONS, TASK_STG]

PromptTemplateError = TemplatePackError
PromptTemplate = TaskTemplate


def load_prompt_pack(prompt_root):
    return load_task_template_pack(
        prompt_root,
        allowed_variables=_ALLOWED_INFERENCE_VARIABLES,
    )


def load_oracle_prompt_pack(prompt_root):
    return {
        variant: load_task_template_pack(
            prompt_root / variant,
            allowed_variables=_ALLOWED_ORACLE_INFERENCE_VARIABLES,
            task_names=_ORACLE_UPSTREAM_TASKS,
        )
        for variant in ORACLE_EXPERIMENT_B_VARIANTS
    }


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
        return 'Return JSON only: {"text": "...", "bbox": [x1, y1, x2, y2]}'
    if task_name == TASK_OBJECTS_SPATIAL:
        return (
            'Return JSON only: {"text": "...", "objects": [{"label": "...", "bbox": '
            '[x1, y1, x2, y2]}, {"label": "...", "bbox": [x1, y1, x2, y2]}]}'
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


def _oracle_output_contract(task_name: str) -> str:
    if task_name == TASK_CONTINUOUS_ACTIONS:
        return 'Return JSON only: {"segments": [{"start_sampled": 0, "end_sampled": 3, "text": "..."}]}'
    if task_name == TASK_STG:
        return 'Return JSON only: {"time_window_sampled": [0, 4]}'
    raise PromptTemplateError(f"unsupported task_name for OracleTrack output contract: {task_name}")


def _sampled_index_range(sample: PreparedSample) -> str:
    if not sample.sampled_frames_original:
        raise PromptTemplateError(f"{sample.sample_id}: no sampled frames available")
    return f"0..{len(sample.sampled_frames_original) - 1}"


def _required_object_labels_json(sample: PreparedSample) -> str:
    if sample.task_name != TASK_OBJECTS_SPATIAL:
        return "[]"
    objects = sample.reference_payload.get("objects")
    if not isinstance(objects, list):
        raise PromptTemplateError(f"{sample.sample_id}: missing objects in reference_payload")
    labels = [str(obj["label"]).strip() for obj in objects]
    if any(not label for label in labels):
        raise PromptTemplateError(f"{sample.sample_id}: object labels must be non-empty")
    return json.dumps(labels, ensure_ascii=False)


def _sampled_video_fps(sample: PreparedSample) -> str:
    if len(sample.sampled_frames_original) <= 1:
        return ""
    if sample.sampled_video_fps is None:
        raise PromptTemplateError(f"{sample.sample_id}: missing sampled_video_fps")
    return f"{sample.sampled_video_fps:.3f}".rstrip("0").rstrip(".")


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
        "num_sampled_frames": len(sample.sampled_frames_original),
        "sampled_index_range": _sampled_index_range(sample),
        "sampled_video_fps": _sampled_video_fps(sample),
        "output_contract": _output_contract(sample.task_name),
        "required_object_labels_json": _required_object_labels_json(sample),
    }
    prompt_text = render_template_text(template.prompt_template, variables, template.path)
    return RenderedPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        prompt_text=prompt_text,
        variables=variables,
    )


def _oracle_tracking_explanation(sample: PreparedSample) -> str:
    if sample.task_name == TASK_STG:
        return (
            "Each row already gives the target subject's known location in the "
            "normalized_1000 coordinate system, where `(0, 0)` is the top-left corner "
            "of the frame and `(1000, 1000)` is the bottom-right corner. Each row uses "
            "`frame_sampled` as a sampled-frame index and `bbox_mot` as "
            "`[left, top, width, height]`. Use these known boxes only to identify the "
            "target subject. You only need to predict `time_window_sampled`."
        )
    if sample.task_name == TASK_CONTINUOUS_ACTIONS:
        return (
            "Each row already gives the target athlete's known location in the "
            "normalized_1000 coordinate system, where `(0, 0)` is the top-left corner "
            "of the frame and `(1000, 1000)` is the bottom-right corner. Each row uses "
            "`frame_sampled` as a sampled-frame index and `bbox_mot` as "
            "`[left, top, width, height]`. Use these known boxes only to identify the "
            "target athlete. You only need to describe action segments."
        )
    raise PromptTemplateError(f"unsupported oracle upstream task: {sample.task_name}")


def _oracle_visual_explanation(sample: PreparedSample) -> str:
    if sample.task_name == TASK_STG:
        return (
            "The target subject has already been highlighted with GT tracking boxes directly "
            "on the sampled inputs. Use the highlighted boxes only to identify the target "
            "subject. You only need to predict `time_window_sampled`."
        )
    if sample.task_name == TASK_CONTINUOUS_ACTIONS:
        return (
            "The target athlete has already been highlighted with GT tracking boxes directly "
            "on the sampled inputs. Use the highlighted boxes only to identify the target "
            "athlete. You only need to describe action segments."
        )
    raise PromptTemplateError(f"unsupported oracle upstream task: {sample.task_name}")


def render_oracle_upstream_prompt(
    prompt_pack: dict[str, dict[str, PromptTemplate]],
    sample: PreparedSample,
    *,
    variant: str,
) -> RenderedPrompt:
    if sample.task_name not in set(_ORACLE_UPSTREAM_TASKS):
        raise PromptTemplateError(
            f"oracle upstream prompt is only supported for {TASK_CONTINUOUS_ACTIONS} and {TASK_STG}"
        )
    if variant not in ORACLE_EXPERIMENT_B_VARIANTS:
        raise PromptTemplateError(f"unsupported Oracle variant: {variant}")
    tracking_gt = sample.reference_payload.get("tracking_gt_sampled")
    if not isinstance(tracking_gt, list):
        raise PromptTemplateError(f"{sample.sample_id}: missing tracking_gt_sampled for OracleTrack")
    try:
        variant_pack = prompt_pack[variant]
        template = variant_pack[sample.task_name]
    except KeyError as exc:
        raise PromptTemplateError(
            f"missing Oracle prompt template for variant {variant} task {sample.task_name}"
        ) from exc
    variables: dict[str, object] = {
        "question": sample.question_text,
        "num_sampled_frames": len(sample.sampled_frames_original),
        "sampled_index_range": _sampled_index_range(sample),
        "sampled_video_fps": _sampled_video_fps(sample),
        "output_contract": _oracle_output_contract(sample.task_name),
        "oracle_tracking_explanation": _oracle_tracking_explanation(sample),
        "oracle_tracking_json": json.dumps(tracking_gt, ensure_ascii=False, indent=2),
        "oracle_visual_explanation": _oracle_visual_explanation(sample),
    }
    prompt_text = render_template_text(template.prompt_template, variables, template.path)
    return RenderedPrompt(
        task_name=sample.task_name,
        template_path=str(template.path),
        prompt_text=prompt_text,
        variables=variables,
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
    upstream_prompt_text: str,
    upstream_raw_output: Any,
) -> list[PromptMessage]:
    return [
        PromptMessage(role="user", content=upstream_prompt_text),
        PromptMessage(role="assistant", content=serialize_model_output(upstream_raw_output)),
    ]
