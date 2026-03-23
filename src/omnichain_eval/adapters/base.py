"""Model adapter interface and a mock implementation."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod

from ..constants import (
    TASK_COMMENTARY,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_SINGLE,
    TASK_STG,
    TEXT_ONLY_TASKS,
)
from ..schema import ModelInput


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
        model_input: ModelInput,
    ) -> str:
        raise NotImplementedError


class MockAdapter(BaseModelAdapter):
    """Returns perfect answers from prepared GT payloads for smoke tests."""

    def supports_oracle_track(self) -> bool:
        return True

    def predict(
        self,
        model_input: ModelInput,
    ) -> str:
        sample = model_input.sample
        oracle_track = model_input.oracle_track
        reference = sample.reference_payload
        payload: dict[str, object]
        if sample.task_name in TEXT_ONLY_TASKS:
            payload = {"text": reference["text"]}
        elif sample.task_name == TASK_SCOREBOARD_SINGLE:
            payload = {"text": reference["text"], "bbox": reference["bbox"]}
        elif sample.task_name == TASK_OBJECTS_SPATIAL:
            payload = {
                "text": reference["text"],
                "bbox_a": reference["bbox_a"],
                "bbox_b": reference["bbox_b"],
            }
        elif sample.task_name in {TASK_CONTINUOUS_EVENTS, TASK_COMMENTARY}:
            payload = {"segments": reference["segments_sampled"]}
        elif sample.task_name == TASK_CONTINUOUS_ACTIONS:
            payload = {
                "segments": reference["segments_sampled"],
                "tracking": reference.get("tracking_gt_sampled", []),
            }
        elif sample.task_name == TASK_STG:
            tracking = [] if oracle_track else reference.get("tracking_gt_sampled", [])
            payload = {
                "time_window_sampled": reference["time_window_sampled"],
                "tracking": tracking,
            }
        else:
            raise ValueError(f"mock adapter does not support task {sample.task_name}")
        return json.dumps(payload, ensure_ascii=False)


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
