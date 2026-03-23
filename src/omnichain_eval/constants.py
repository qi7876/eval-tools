"""Benchmark-wide constants and task definitions."""

from __future__ import annotations

VIDEO_FPS = 10
VIDEO_RESOLUTION = (1920, 1080)
MAIN_FRAME_BUDGET = 64
SHORT_WINDOW_FRAMES = 320
STG_EXPANSION_FRAMES = 50
BBOX_IOU_THRESHOLD = 0.5
TIOU_THRESHOLD = 0.5
TRACKING_THRESHOLD = 0.5

TASK_SCOREBOARD_SINGLE = "Scoreboard_Single"
TASK_OBJECTS_SPATIAL = "Objects_Spatial_Relationships"
TASK_SCOREBOARD_MULTIPLE = "Scoreboard_Multiple"
TASK_CONTINUOUS_ACTIONS = "Continuous_Actions_Caption"
TASK_CONTINUOUS_EVENTS = "Continuous_Events_Caption"
TASK_STG = "Spatial_Temporal_Grounding"
TASK_SPATIAL_IMAGINATION = "Spatial_Imagination"
TASK_TEMPORAL_CAUSAL = "Temporal_Causal"
TASK_SCORE_PREDICTION = "Score_Prediction"
TASK_AI_COACH = "AI_Coach"

ALL_TASKS = {
    TASK_SCOREBOARD_SINGLE,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_MULTIPLE,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_STG,
    TASK_SPATIAL_IMAGINATION,
    TASK_TEMPORAL_CAUSAL,
    TASK_SCORE_PREDICTION,
    TASK_AI_COACH,
}

SINGLE_FRAME_TASKS = {
    TASK_SCOREBOARD_SINGLE,
    TASK_OBJECTS_SPATIAL,
}

WINDOW_TASKS = {
    TASK_SCOREBOARD_MULTIPLE,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_SPATIAL_IMAGINATION,
    TASK_TEMPORAL_CAUSAL,
    TASK_SCORE_PREDICTION,
    TASK_AI_COACH,
}

SEGMENT_TASKS = {
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
}

TEXT_ONLY_TASKS = {
    TASK_SCOREBOARD_MULTIPLE,
    TASK_SPATIAL_IMAGINATION,
    TASK_TEMPORAL_CAUSAL,
    TASK_SCORE_PREDICTION,
    TASK_AI_COACH,
}

JUDGE_REQUIRED_TASKS = {
    TASK_SCOREBOARD_SINGLE,
    TASK_OBJECTS_SPATIAL,
    TASK_SCOREBOARD_MULTIPLE,
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_SPATIAL_IMAGINATION,
    TASK_TEMPORAL_CAUSAL,
    TASK_SCORE_PREDICTION,
    TASK_AI_COACH,
}

EXPERIMENT_D_TASKS = WINDOW_TASKS.copy()

STG_UPSTREAM_TASKS = {
    TASK_CONTINUOUS_ACTIONS,
    TASK_STG,
}

STRUCTURER_MODEL_DEFAULT = "deepseek-ai/DeepSeek-V3.2"
STRUCTURER_SYSTEM_PROMPT = """You are a structured extraction assistant for a video understanding benchmark.

Your job is to convert a model's raw answer into the benchmark's canonical JSON format.

Important:
- Use only information present in the raw model output.
- Do not infer missing coordinates, intervals, or tracking rows.
- If a field is missing, leave it empty using the task's canonical empty value.
- Return JSON only.
"""

JUDGE_MODEL_DEFAULT = "deepseek-ai/DeepSeek-V3.2"
JUDGE_JSON_KEYS = {
    "correctness",
    "completeness",
    "faithfulness",
    "final_pass",
    "confidence",
    "brief_reason",
}

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
