"""Model adapter interface and a mock implementation."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from ..constants import (
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_SINGLE,
    TASK_STG,
    TEXT_ONLY_TASKS,
)
from ..schema import PreparedSample


class BaseModelAdapter(ABC):
    """Base class for model-specific inference adapters."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def supports_commentary(self) -> bool:
        return True

    def supports_oracle_track(self) -> bool:
        return False

    @abstractmethod
    def predict(
        self,
        sample: PreparedSample,
        *,
        oracle_track: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError


class MockAdapter(BaseModelAdapter):
    """Returns perfect answers from prepared GT payloads for smoke tests."""

    def supports_oracle_track(self) -> bool:
        return True

    def predict(
        self,
        sample: PreparedSample,
        *,
        oracle_track: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Any:
        reference = sample.reference_payload
        if sample.task_name in TEXT_ONLY_TASKS:
            return {"text": reference["text"]}
        if sample.task_name == TASK_SCOREBOARD_SINGLE:
            return {"text": reference["text"], "bbox": reference["bbox"]}
        if sample.task_name == TASK_OBJECTS_SPATIAL:
            return {
                "text": reference["text"],
                "bbox_a": reference["bbox_a"],
                "bbox_b": reference["bbox_b"],
            }
        if sample.task_name in {TASK_CONTINUOUS_EVENTS, TASK_COMMENTARY}:
            return {"segments": reference["segments_sampled"]}
        if sample.task_name == TASK_CONTINUOUS_ACTIONS:
            return {
                "segments": reference["segments_sampled"],
                "tracking": reference.get("tracking_gt_sampled", []),
            }
        if sample.task_name == TASK_STG:
            tracking = [] if oracle_track else reference.get("tracking_gt_sampled", [])
            return {
                "time_window_sampled": reference["time_window_sampled"],
                "tracking": tracking,
            }
        raise ValueError(f"mock adapter does not support task {sample.task_name}")


def resolve_adapter(spec: str) -> BaseModelAdapter:
    if spec == "mock":
        return MockAdapter()
    if ":" not in spec:
        raise ValueError("adapter spec must be 'mock' or 'module.path:ClassName'")
    module_name, class_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls()
    if not isinstance(adapter, BaseModelAdapter):
        raise TypeError(f"{spec} did not resolve to a BaseModelAdapter instance")
    return adapter
