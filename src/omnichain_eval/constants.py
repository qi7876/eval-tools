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
TASK_COMMENTARY = "Commentary"

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
    TASK_COMMENTARY,
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
    TASK_COMMENTARY,
}

SEGMENT_TASKS = {
    TASK_CONTINUOUS_ACTIONS,
    TASK_CONTINUOUS_EVENTS,
    TASK_COMMENTARY,
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
    TASK_COMMENTARY,
}

EXPERIMENT_D_TASKS = WINDOW_TASKS.copy()

TASKS_EXCLUDED_FROM_OVERALL = {TASK_COMMENTARY}

STG_UPSTREAM_TASKS = {
    TASK_CONTINUOUS_ACTIONS,
    TASK_STG,
}

JUDGE_MODEL_DEFAULT = "deepseek-ai/DeepSeek-V3.2"
JUDGE_SYSTEM_PROMPT = """You are an evaluator for a video understanding benchmark.

Your job is to judge whether the model prediction is acceptable given:
1. the task definition,
2. the question,
3. the reference answer,
4. the model prediction.

Rubric:
- Correctness: Is the prediction factually consistent with the reference?
- Completeness: Does it answer the core question sufficiently?
- Faithfulness: Does it avoid unsupported details or contradictions?

Important:
- Minor wording differences are allowed.
- Paraphrases are allowed.
- If the prediction misses a key entity, relation, direction, cause, or outcome, it should fail.
- Do not reward verbosity.
- Return JSON only.
"""

JUDGE_JSON_KEYS = {
    "correctness",
    "completeness",
    "faithfulness",
    "final_pass",
    "confidence",
    "brief_reason",
}

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

