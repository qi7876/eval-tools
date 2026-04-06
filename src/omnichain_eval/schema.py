"""Typed records used across the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import to_jsonable


@dataclass(slots=True)
class VideoMetadata:
    duration_sec: float
    fps: int
    total_frames: int
    resolution: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    annotation_id: str
    video_key: str
    task_name: str
    task_level: str
    question_text: str
    source_annotation_path: Path
    source_video_path: Path
    video_metadata: VideoMetadata
    raw_annotation: dict[str, Any]
    reference_payload: dict[str, Any]
    timestamp_frame: int | None = None
    q_window: tuple[int, int] | None = None
    a_window: tuple[int, int] | None = None
    source_tracking_path: Path | None = None
    upstream_annotation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class ProtocolSpec:
    protocol_id: str
    description: str
    frame_budget: int

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class PreparedSample:
    sample_id: str
    annotation_id: str
    video_key: str
    task_name: str
    task_level: str
    protocol_id: str
    question_text: str
    sampled_frames_original: list[int]
    sampled_to_original: dict[int, int]
    frame_files: list[str]
    source_video_path: str
    source_annotation_path: str
    reference_payload: dict[str, Any]
    timestamp_frame: int | None = None
    q_window: tuple[int, int] | None = None
    q_window_sampled: tuple[int, int] | None = None
    a_window: tuple[int, int] | None = None
    source_tracking_path: str | None = None
    upstream_annotation_id: str | None = None
    sampled_video_file: str | None = None
    sampled_video_fps: float | None = None
    oracle_visual_frame_files: list[str] = field(default_factory=list)
    oracle_visual_sampled_video_file: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparedSample":
        sampled_to_original = {
            int(key): int(value) for key, value in payload["sampled_to_original"].items()
        }
        return cls(
            sample_id=payload["sample_id"],
            annotation_id=payload["annotation_id"],
            video_key=payload["video_key"],
            task_name=payload["task_name"],
            task_level=payload["task_level"],
            protocol_id=payload["protocol_id"],
            question_text=payload["question_text"],
            sampled_frames_original=[int(value) for value in payload["sampled_frames_original"]],
            sampled_to_original=sampled_to_original,
            frame_files=list(payload["frame_files"]),
            source_video_path=payload["source_video_path"],
            source_annotation_path=payload["source_annotation_path"],
            reference_payload=payload["reference_payload"],
            timestamp_frame=payload.get("timestamp_frame"),
            q_window=tuple(payload["q_window"]) if payload.get("q_window") else None,
            q_window_sampled=(
                tuple(payload["q_window_sampled"]) if payload.get("q_window_sampled") else None
            ),
            a_window=tuple(payload["a_window"]) if payload.get("a_window") else None,
            source_tracking_path=payload.get("source_tracking_path"),
            upstream_annotation_id=payload.get("upstream_annotation_id"),
            sampled_video_file=payload.get("sampled_video_file"),
            sampled_video_fps=(
                float(payload["sampled_video_fps"])
                if payload.get("sampled_video_fps") is not None
                else None
            ),
            oracle_visual_frame_files=list(payload.get("oracle_visual_frame_files", [])),
            oracle_visual_sampled_video_file=payload.get("oracle_visual_sampled_video_file"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class PromptMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class RenderedPrompt:
    task_name: str
    template_path: str
    prompt_text: str
    variables: dict[str, Any]


@dataclass(slots=True)
class ModelInput:
    sample: PreparedSample
    messages: list[PromptMessage]

    def messages_as_dicts(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]


@dataclass(slots=True)
class StructuredPredictionResult:
    task_name: str
    raw_output: str
    structured_prediction: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structurer_raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class StructuredPredictionRecord:
    sample_id: str
    task_name: str
    video_key: str
    protocol_id: str
    raw_output: str
    structured_prediction: dict[str, Any] | None
    structuring_errors: list[str]
    structuring_warnings: list[str]
    structurer_raw_response: str | None = None
    pair_id: str | None = None
    upstream_sample_id: str | None = None
    downstream_sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredPredictionRecord":
        return cls(
            sample_id=payload["sample_id"],
            task_name=payload["task_name"],
            video_key=payload["video_key"],
            protocol_id=payload["protocol_id"],
            raw_output=str(payload["raw_output"]),
            structured_prediction=payload.get("structured_prediction"),
            structuring_errors=list(payload.get("structuring_errors", [])),
            structuring_warnings=list(payload.get("structuring_warnings", [])),
            structurer_raw_response=payload.get("structurer_raw_response"),
            pair_id=payload.get("pair_id"),
            upstream_sample_id=payload.get("upstream_sample_id"),
            downstream_sample_id=payload.get("downstream_sample_id"),
        )


@dataclass(slots=True)
class JudgeDecision:
    correctness: int
    completeness: int
    faithfulness: int
    final_pass: int
    confidence: str
    brief_reason: str
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class EvaluationRecord:
    sample_id: str
    task_name: str
    video_key: str
    protocol_id: str
    structured_prediction: dict[str, Any] | None
    structuring_errors: list[str]
    structuring_warnings: list[str]
    component_metrics: dict[str, Any]
    component_pass: dict[str, Any]
    task_pass: int
    judge_decision: dict[str, Any] | None = None
    raw_output: str | None = None
    bertscore_candidate: str | None = None
    bertscore_reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationRecord":
        return cls(
            sample_id=payload["sample_id"],
            task_name=payload["task_name"],
            video_key=payload["video_key"],
            protocol_id=payload["protocol_id"],
            structured_prediction=payload.get("structured_prediction"),
            structuring_errors=list(payload.get("structuring_errors", [])),
            structuring_warnings=list(payload.get("structuring_warnings", [])),
            component_metrics=dict(payload.get("component_metrics", {})),
            component_pass=dict(payload.get("component_pass", {})),
            task_pass=int(payload["task_pass"]),
            judge_decision=payload.get("judge_decision"),
            raw_output=payload.get("raw_output"),
            bertscore_candidate=payload.get("bertscore_candidate"),
            bertscore_reference=payload.get("bertscore_reference"),
        )


@dataclass(slots=True)
class TaskSummary:
    model_name: str
    task_name: str
    protocol_id: str
    num_samples: int
    task_accuracy: float | None
    num_scored_samples: int = 0
    num_pending_samples: int = 0
    judge_pass_rate: float | None = None
    bbox_pass_rate: float | None = None
    tiou_pass_rate: float | None = None
    tracking_mean_iou_mean: float | None = None
    tracking_pass_rate_mean: float | None = None
    mean_bertscore_f1: float | None = None
    median_bertscore_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(slots=True)
class ChainPairRecord:
    pair_id: str
    video_key: str
    upstream_sample_id: str
    downstream_sample_id: str
    upstream_task_name: str

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
