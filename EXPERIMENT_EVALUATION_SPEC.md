# OmniChainBench Experiment Evaluation Specification

## 1. Purpose

This document is the implementation-facing specification for the evaluation pipeline used in the experiments of OmniChainBench. It is written for a coding agent that will implement data preparation, model-output normalization, metric computation, and experiment reporting.

All rules in this document are normative unless a paragraph is explicitly marked as a recommendation or a placeholder. If an implementation detail is not covered here, the evaluator must not invent benchmark logic silently; it should surface the gap as an engineering TODO.

## 2. Global Constants and Conventions

### 2.1 Dataset-Level Constants

- Video FPS is fixed to $10$.
- Video resolution is fixed to $1920 \times 1080$.
- Audio is disabled for evaluation.
- All frame indices are `0-based`.
- All frame intervals are closed intervals of the form $[s, e]$ with $s \le e$.
- All time-window calculations are performed in frame space, not in seconds.

Useful constants:

$$
\text{FPS}_{video} = 10
$$

$$
32s = 320 \text{ frames}, \qquad 5s = 50 \text{ frames}
$$

For any closed interval $I = [s, e]$, its length in frames is

$$
|I| = e - s + 1
$$

### 2.2 Two Index Spaces

The evaluator must distinguish between two frame-index spaces:

1. `original frame index`
   This is the frame index in the original video.
2. `sampled frame index`
   This is the frame index in the sampled model input sequence after frame selection.

The model must express all predicted temporal outputs using `sampled frame index`.

The evaluator must always construct and persist a mapping

$$
M: i_{\text{sampled}} \mapsto i_{\text{original}}
$$

where $M(k)$ is the original frame index corresponding to sampled position $k$.

This mapping is mandatory for:

- converting temporal predictions back to original-frame coordinates,
- locating GT tracking annotations on sampled frames,
- computing STG tracking metrics,
- debugging model outputs.

### 2.3 Interval Clipping

Whenever a computed interval goes out of video range, it must be clipped to $[0, T-1]$, where $T$ is the total number of frames in the video:

$$
\operatorname{clip}(x; 0, T-1) = \min(\max(x, 0), T-1)
$$

For an interval $[s, e]$, clip both endpoints independently and then ensure $s \le e$.

## 3. Task Inventory and Required Outputs

The benchmark contains 11 tasks. Commentary is evaluated separately from the other 10 tasks in overall reporting.

### 3.1 Single-Frame Tasks

These tasks consume exactly one original frame and do not use temporal-window expansion.

- `Scoreboard_Single`
  Required output: text answer + one spatial box.
- `Objects_Spatial_Relationships`
  Required output: text answer + two spatial boxes.

### 3.2 Window-Based Tasks

These tasks use a question interval `Q_window_frame = [q_s, q_e]` unless otherwise stated.

- `Scoreboard_Multiple`
  Required output: text answer.
- `Continuous_Actions_Caption`
  Required output: time-aligned action captions + tracking boxes.
- `Continuous_Events_Caption`
  Required output: time-aligned event captions.
- `Spatial_Imagination`
  Required output: text answer.
- `Temporal_Causal`
  Required output: text answer.
- `Score_Prediction`
  Required output: text answer.
- `AI_Coach`
  Required output: text answer.
- `Commentary`
  Required output: time-aligned commentary.

### 3.3 Special Task: Spatio-Temporal Grounding

`Spatial_Temporal_Grounding` uses a task-specific input rule. The model predicts:

- one temporal window,
- one single-object tracking result.

Its GT answer interval is `A_window_frame = [a_s, a_e]`.

## 4. Canonical Prediction Normalization

Before scoring, raw model outputs should be normalized into task-specific structured objects. The exact external model prompt can vary, but the evaluator should transform outputs into the following canonical forms.

### 4.1 Text-Only Tasks

```json
{
  "text": "..."
}
```

Applicable to:

- `Scoreboard_Multiple`
- `Spatial_Imagination`
- `Temporal_Causal`
- `Score_Prediction`
- `AI_Coach`

### 4.2 Single-Frame Mixed Tasks

`Scoreboard_Single`:

```json
{
  "text": "...",
  "bbox": [xtl, ytl, xbr, ybr]
}
```

`Objects_Spatial_Relationships`:

```json
{
  "text": "...",
  "objects": [
    {"label": "Player A", "bbox": [xtl, ytl, xbr, ybr]},
    {"label": "Player B", "bbox": [xtl, ytl, xbr, ybr]}
  ]
}
```

Single-frame spatial boxes use the stored file convention:

$$
[xtl, ytl, xbr, ybr]
$$

### 4.3 Segment-Based Text Tasks

For tasks whose answers are sequences of segments, normalize to:

```json
{
  "segments": [
    {
      "start_sampled": 0,
      "end_sampled": 3,
      "text": "..."
    }
  ]
}
```

Applicable to:

- `Continuous_Events_Caption`
- `Commentary`

For `Continuous_Actions_Caption`, use:

```json
{
  "segments": [
    {
      "start_sampled": 0,
      "end_sampled": 3,
      "text": "..."
    }
  ],
  "tracking": [
    {
      "frame_sampled": 0,
      "bbox_mot": [left, top, width, height]
    }
  ]
}
```

### 4.4 Spatio-Temporal Grounding

```json
{
  "time_window_sampled": [start_sampled, end_sampled],
  "tracking": [
    {
      "frame_sampled": 0,
      "bbox_mot": [left, top, width, height]
    }
  ]
}
```

Tracking outputs use MOT-style geometry fields:

$$
[left, top, width, height]
$$

If a MOT-like row contains a `track_id`, it must be ignored in scoring. The benchmark is single-object for tracking evaluation.

## 5. Main Input Protocol

### 5.1 High-Level Rule

The main benchmark protocol fixes the frame budget at

$$
B = 64
$$

and strictly forbids future information beyond the query horizon.

The protocol does not define a separate universal history window. Instead, sampling depends on task type and query duration.

### 5.2 Single-Frame Tasks

For single-frame tasks with `timestamp_frame = t`:

- input consists of exactly one original frame $t$,
- no history expansion is performed,
- no temporal sampling is performed.

Therefore:

$$
S = 1, \qquad M(0) = t
$$

where $S$ is the sampled sequence length.

### 5.3 Window Tasks with `Q_window_frame`

Let the question interval be

$$
Q = [q_s, q_e]
$$

### Case A: Short Query Interval

If

$$
|Q| \le 320
$$

the evaluator uses the query end frame $q_e$ as the anchor and samples backwards at nominal $2$ fps from the most recent history. In frame space, $2$ fps at $10$ fps video corresponds to stride $5$.

Define the visible interval:

$$
V = [\max(0, q_e - 319), q_e]
$$

Construct candidate sampled frames backwards:

$$
c_k = q_e - 5k, \qquad k = 0, 1, \dots, 63
$$

Keep only candidates such that $c_k \in V$. Reverse them into chronological order. The resulting sampled sequence may contain fewer than $64$ frames near the beginning of a video.

Important:

- do not refill missing early-history frames with duplicates,
- do not invent extra padding as evidence,
- implementation-level tensor padding is allowed later, but padded frames do not count as observed evidence.

### Case B: Long Query Interval

If

$$
|Q| > 320
$$

the evaluator samples $64$ frames uniformly from the full question interval.

Let

$$
V = [q_s, q_e]
$$

Generate $64$ floating-point sample positions:

$$
u_k = q_s + \frac{k}{63}(q_e - q_s), \qquad k = 0, 1, \dots, 63
$$

Convert each $u_k$ to the nearest original frame index:

$$
\hat{u}_k = \operatorname{round}(u_k)
$$

Clip $\hat{u}_k$ into $V$ if needed, remove duplicates while preserving order, and assign sampled indices in chronological order.

This is the implementation form of the adaptive-FPS rule:

$$
f_{\mathrm{eff}} = \min\left(2, \frac{B}{L}\right)
$$

where $L$ is the visible query duration in seconds.

### 5.4 Special Input Rule for `Spatial_Temporal_Grounding`

`Spatial_Temporal_Grounding` does not use the standard `Q_window_frame` rule in the main protocol.

Instead, let the GT answer interval be

$$
A = [a_s, a_e]
$$

Construct the STG input interval by expanding the GT answer interval by $5$ seconds on both sides:

$$
V_{\mathrm{STG}} = [a_s - 50, a_e + 50]
$$

Then clip the interval to video bounds:

$$
V_{\mathrm{STG}} = [\operatorname{clip}(a_s - 50), \operatorname{clip}(a_e + 50)]
$$

Sample $64$ frames uniformly from this interval using the same uniform rule as in the long-query case:

$$
u_k = s + \frac{k}{63}(e - s), \qquad k = 0, 1, \dots, 63
$$

where $[s, e] = V_{\mathrm{STG}}$.

`Spatial_Temporal_Grounding` does not participate in Experiment D ablations.

### 5.5 Sampled-to-Original Mapping

For every evaluated sample, persist:

- `sampled_frames_original`: ordered list of original frame indices,
- `sampled_to_original`: identical information in mapping form,
- optionally `original_to_sampled` if needed by helper utilities.

All temporal predictions must first be interpreted in sampled space and then mapped back to original space through $M$ before final metric computation.

If a predicted sampled interval is

$$
[\hat{s}, \hat{e}]
$$

its corresponding original-frame interval is

$$
[M(\hat{s}), M(\hat{e})]
$$

assuming $\hat{s}$ and $\hat{e}$ are valid sampled indices and $\hat{s} \le \hat{e}$.

## 6. Geometry and Temporal Conventions

### 6.1 Spatial Boxes

Single-frame task boxes are stored and scored as:

$$
[xtl, ytl, xbr, ybr]
$$

Tracking boxes are stored and scored in MOT-style geometry:

$$
[left, top, width, height]
$$

For IoU computation, convert MOT boxes to corner form:

$$
x_1 = left,\quad y_1 = top,\quad x_2 = left + width,\quad y_2 = top + height
$$

### 6.2 Spatial IoU

For two boxes $B_p$ and $B_g$ in corner form:

$$
B = [x_1, y_1, x_2, y_2]
$$

the intersection width and height are

$$
w_{\cap} = \max(0, \min(x_2^p, x_2^g) - \max(x_1^p, x_1^g))
$$

$$
h_{\cap} = \max(0, \min(y_2^p, y_2^g) - \max(y_1^p, y_1^g))
$$

the intersection area is

$$
A_{\cap} = w_{\cap} \cdot h_{\cap}
$$

and IoU is

$$
\operatorname{IoU}(B_p, B_g) =
\frac{A_{\cap}}{A_p + A_g - A_{\cap}}
$$

The spatial pass threshold is fixed to:

$$
\operatorname{IoU} \ge 0.5
$$

### 6.3 Temporal IoU

For original-frame closed intervals

$$
I_p = [s_p, e_p], \qquad I_g = [s_g, e_g]
$$

temporal intersection length is

$$
|I_p \cap I_g| = \max(0, \min(e_p, e_g) - \max(s_p, s_g) + 1)
$$

temporal union length is

$$
|I_p \cup I_g| = |I_p| + |I_g| - |I_p \cap I_g|
$$

and temporal IoU is

$$
\operatorname{tIoU}(I_p, I_g) = \frac{|I_p \cap I_g|}{|I_p \cup I_g|}
$$

The temporal pass threshold is fixed to:

$$
\operatorname{tIoU} \ge 0.5
$$

## 7. Tracking Evaluation

### 7.1 Shared Rule

Tracking is evaluated only for:

- `Continuous_Actions_Caption`
- `Spatial_Temporal_Grounding`

In both tasks, tracking is single-object.

For every GT-annotated sampled frame, there is at most one relevant GT target box. `track_id` is ignored.

### 7.2 Mapping Predicted Tracking to Original Frames

For each predicted tracking record with sampled frame index $k$, map it to original frame index $M(k)$.

If the raw prediction contains multiple boxes for the same sampled frame, the evaluator should collapse them to one box for scoring. Recommended engineering rule: keep the first parsed box and log a warning. This is an implementation safeguard, not a scientific metric choice.

### 7.3 `Continuous_Actions_Caption` Tracking Metric

For `Continuous_Actions_Caption`, evaluate tracking only on sampled frames that have GT tracking annotations.

For each such sampled frame:

- if the model predicts a box for that sampled frame, compute IoU against the GT box,
- if the model does not predict a box for that sampled frame, define IoU as $0$.

Let the evaluated sampled-frame set be $\mathcal{F}$ and the IoU for frame $f$ be $\operatorname{IoU}_f$.

Then:

$$
\text{tracking\_mean\_iou} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \operatorname{IoU}_f
$$

$$
\text{tracking\_pass\_rate} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \mathbf{1}[\operatorname{IoU}_f \ge 0.5]
$$

Tracking passes if and only if

$$
\text{tracking\_mean\_iou} \ge 0.5
\quad \text{and} \quad
\text{tracking\_pass\_rate} \ge 0.5
$$

### 7.4 `Spatial_Temporal_Grounding` Tracking Metric

`Spatial_Temporal_Grounding` is special.

First evaluate the predicted time window separately using tIoU on original-frame intervals.

For tracking evaluation only, clamp the model’s temporal support to the GT STG answer interval. Operationally:

1. map predicted sampled-frame references to original-frame indices via $M$,
2. restrict the tracking evaluation to the GT answer interval $[a_s, a_e]$,
3. within that interval, evaluate only sampled frames that have GT tracking annotations,
4. if the model has no predicted box on a required sampled frame, its IoU is $0$.

This means STG tracking is scored inside the correct answer interval, while the predicted time window itself is still scored independently by tIoU.

As in `Continuous_Actions_Caption`:

$$
\text{tracking\_mean\_iou} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \operatorname{IoU}_f
$$

$$
\text{tracking\_pass\_rate} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \mathbf{1}[\operatorname{IoU}_f \ge 0.5]
$$

Tracking passes if and only if both thresholds are met:

$$
\text{tracking\_mean\_iou} \ge 0.5
\quad \text{and} \quad
\text{tracking\_pass\_rate} \ge 0.5
$$

## 8. LLM-as-a-Judge

### 8.1 Judge Model Configuration

Use:

- model: `deepseek-ai/DeepSeek-V3.2`
- `temperature = 0`
- `top_p = 1.0`
- `top_k = 1` when supported; otherwise greedy decoding
- `max_new_tokens = 256`
- `n = 1`
- `seed = 42` when supported

Judge output must be structured JSON only.

### 8.2 Base Judge Prompt

The shared base prompt is:

```text
You are an evaluator for a video understanding benchmark.

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
```

### 8.3 JSON Schema

The required judge JSON schema is:

```json
{
  "correctness": 1,
  "completeness": 1,
  "faithfulness": 1,
  "final_pass": 1,
  "confidence": "high",
  "brief_reason": "Prediction matches the reference answer with acceptable paraphrasing."
}
```

Rules:

- `correctness`, `completeness`, `faithfulness` are binary `0/1`.
- `final_pass` is binary `0/1`.
- `brief_reason` should be one sentence.

If parsing the judge JSON fails, the sample must be marked as judge-fail unless a deterministic retry policy is explicitly implemented.

### 8.4 Task-Specific Judge Logic

### A. Closed Factual Text Tasks

Applicable to:

- `Scoreboard_Single` text field
- `Scoreboard_Multiple`
- `Objects_Spatial_Relationships` text field
- `Spatial_Imagination`
- `Score_Prediction`

Rule:

- all core slots must be correct,
- key entity, direction, score, rank, or relation errors cause failure,
- paraphrase is allowed.

Pass condition:

$$
\text{correctness} = 1 \land \text{completeness} = 1 \land \text{faithfulness} = 1
$$

### B. `Temporal_Causal`

Rule:

- pass if the main cause is correct and there is no key hallucination,
- wording can differ,
- minor omitted secondary details do not matter,
- the main cause must not be replaced by a side effect.

Implementation rule:

$$
\text{final\_pass} = 1
$$

if and only if the main cause is correct and faithfulness is preserved. This is intentionally less strict than requiring all three binary fields to be `1`.

### C. Segment-Based Text Tasks

Applicable to:

- `Continuous_Actions_Caption`
- `Continuous_Events_Caption`
- `Commentary`

These tasks do not use separate temporal IoU for segment timing. Instead, `time alignment + textual content` are judged jointly by the LLM judge.

Before calling the judge:

1. convert each predicted sampled interval to an original-frame interval using $M$,
2. convert GT segment intervals into the same original-frame coordinate space,
3. provide both reference and prediction as ordered segment lists.

Recommended normalized judge payload:

```json
{
  "reference_segments": [
    {"start_frame": 100, "end_frame": 130, "text": "..."}
  ],
  "prediction_segments": [
    {"start_frame": 102, "end_frame": 128, "text": "..."}
  ]
}
```

Task-specific rubric for the judge:

- key events/actions must be covered,
- temporal order must be correct,
- rough alignment must be reasonable,
- hallucinated key events or meaning-changing temporal mistakes cause failure.

For these tasks, `final_pass` from the judge is the text/time pass indicator.

### D. `AI_Coach`

Rule:

- multiple valid answers may exist,
- pass if the suggestion is relevant, actionable, and aligned with the reference intent,
- fail if it is generic, irrelevant, or contradicts the situation.

### 8.5 One-Vote Veto

Regardless of task category, the following errors force failure:

- key entity error,
- left/right inversion,
- causal inversion,
- wrong score or ranking,
- obvious hallucination that changes meaning.

## 9. BERTScore

BERTScore is supplementary only. It never determines pass/fail.

Use:

- package: `bert-score`
- model: `microsoft/deberta-xlarge-mnli`
- report: `F1`
- `rescale_with_baseline=True`
- `idf=False`
- `use_fast_tokenizer=False`

Reference implementation:

```python
from bert_score import score

P, R, F1 = score(
    cands=predictions,
    refs=references,
    model_type="microsoft/deberta-xlarge-mnli",
    lang="en",
    rescale_with_baseline=True,
    idf=False,
    use_fast_tokenizer=False,
)
```

Compute BERTScore only for English text tasks.

Recommended reporting:

- sample-level `bertscore_f1`,
- task-level `mean_bertscore_f1`,
- task-level `median_bertscore_f1`.

For segment-based English tasks, a practical linearization rule is to concatenate segment texts in temporal order. This remains supplementary and does not affect gating.

## 10. Task-Level Main Metrics

For every task, the main reported task score is sample accuracy:

$$
\text{TaskAcc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{sample } i \text{ passes}]
$$

where the sample pass indicator is task-specific.

### 10.1 `Scoreboard_Single`

Sample passes iff:

- judge text pass is `1`,
- bbox IoU $\ge 0.5$.

### 10.2 `Scoreboard_Multiple`

Sample passes iff judge text pass is `1`.

### 10.3 `Objects_Spatial_Relationships`

Sample passes iff:

- judge text pass is `1`,
- the predicted object with the same `label` as GT object 1 has IoU $\ge 0.5$,
- the predicted object with the same `label` as GT object 2 has IoU $\ge 0.5$.

Both boxes must individually satisfy the threshold. Averaging the two IoUs is not allowed.

### 10.4 `Continuous_Actions_Caption`

Sample passes iff:

- segment text/time judge pass is `1`,
- tracking pass is `1`.

Formally:

$$
\text{Pass}_{CAC} = \text{JudgePass}_{CAC} \cdot \text{TrackingPass}_{CAC}
$$

### 10.5 `Continuous_Events_Caption`

Sample passes iff segment text/time judge pass is `1`.

### 10.6 `Spatial_Temporal_Grounding`

Sample passes iff:

- temporal window tIoU pass is `1`,
- tracking pass is `1`.

Formally:

$$
\text{Pass}_{STG} = \mathbf{1}[\operatorname{tIoU} \ge 0.5] \cdot \text{TrackingPass}_{STG}
$$

### 10.7 `Spatial_Imagination`

Sample passes iff judge text pass is `1`.

### 10.8 `Temporal_Causal`

Sample passes iff the judge determines that the main cause is correct and there is no key hallucination.

### 10.9 `Score_Prediction`

Sample passes iff judge text pass is `1`.

### 10.10 `AI_Coach`

Sample passes iff judge text pass is `1`.

### 10.11 `Commentary`

Sample passes iff segment text/time judge pass is `1`.

This task is reported separately and excluded from overall macro averages.

## 11. Aggregate Reporting

### 11.1 Per-Task Reporting

Each task must report its own main task accuracy. Where relevant, also report component metrics:

- bbox IoU pass rate,
- tIoU pass rate,
- tracking mean IoU,
- tracking frame pass rate,
- judge pass rate,
- BERTScore mean/median if applicable.

### 11.2 Overall Metric

The main `Overall` score for Experiment A is a simple macro average over 10 tasks, excluding `Commentary`.

Let the 10 non-commentary task accuracies be $a_1, \dots, a_{10}$. Then:

$$
\text{Overall}_{10\text{ tasks}} = \frac{1}{10} \sum_{j=1}^{10} a_j
$$

`Commentary` must be reported separately and must not be included in `Overall`.

For models that do not support `Commentary`, report `N/A` for that task rather than `0`. This does not affect `Overall` because commentary is excluded from the main macro average.

## 12. Experiment A

Experiment A is the main benchmark result.

Required scope:

- all 10 baseline models,
- all 11 tasks,
- main protocol only, not model-native.

Required outputs:

- `Overall` score over 10 tasks excluding commentary,
- per-task scores for all 11 tasks,
- commentary reported separately.

The model count is fixed to 10.

## 13. Experiment B: Chain Evaluation

### 13.1 Chain Scope

Experiment B uses the linked subset:

- upstream understanding task is either `Continuous_Actions_Caption` or `Spatial_Temporal_Grounding`,
- downstream reasoning task is `Spatial_Imagination`.

The dataset loader must provide explicit chain pairing metadata. The evaluator must not infer pairs heuristically.

### 13.2 Chain Indicators

For chain instance $i$:

- let $P_i$ be the tracking pass indicator,
- let $U_i$ be the non-tracking understanding pass indicator,
- let $R_i$ be the downstream reasoning pass indicator.

For `Continuous_Actions_Caption`:

$$
U_i = \text{JudgePass}_{CAC,i}
$$

For `Spatial_Temporal_Grounding`:

$$
U_i = \mathbf{1}[\operatorname{tIoU}_i \ge 0.5]
$$

Tracking pass is:

$$
P_i = \mathbf{1}[\text{tracking\_mean\_iou}_i \ge 0.5] \cdot
\mathbf{1}[\text{tracking\_pass\_rate}_i \ge 0.5]
$$

Reasoning pass is:

$$
R_i = \text{JudgePass}_{SpatialImagination,i}
$$

Full understanding pass is:

$$
H_i = P_i \cdot U_i
$$

Chain success is:

$$
CS_i = H_i \cdot R_i = P_i \cdot U_i \cdot R_i
$$

Chain success without track is:

$$
CS_i^{wo} = U_i \cdot R_i
$$

Benchmark-level metrics:

$$
\text{UnderstandingAcc} = \frac{1}{N} \sum_{i=1}^{N} H_i
$$

$$
\text{ReasoningAcc} = \frac{1}{N} \sum_{i=1}^{N} R_i
$$

$$
\text{ChainSuccess} = \frac{1}{N} \sum_{i=1}^{N} CS_i
$$

$$
\text{ChainSuccessWoTrack} = \frac{1}{N} \sum_{i=1}^{N} CS_i^{wo}
$$

### 13.3 OracleTrack

In OracleTrack evaluation:

- replace predicted tracking with GT tracking,
- rerun the affected understanding evaluation and the downstream reasoning evaluation,
- the model does not need to output tracking boxes in the rerun,
- for STG, only tracking is replaced; time-window prediction remains the model’s own prediction.

This is a real rerun of downstream evaluation, not a post-hoc metric recomputation on the original outputs.

Define oracle non-tracking understanding pass as

$$
U_i^{oracle}
$$

and oracle reasoning pass as

$$
R_i^{oracle}
$$

Then:

$$
CS_i^{wo,oracle} = U_i^{oracle} \cdot R_i^{oracle}
$$

and:

$$
\text{UnderstandingAcc@OracleTrack} = \frac{1}{N} \sum_{i=1}^{N} U_i^{oracle}
$$

$$
\text{ReasoningAcc@OracleTrack} = \frac{1}{N} \sum_{i=1}^{N} R_i^{oracle}
$$

$$
\text{ChainSuccessWoTrack@OracleTrack} = \frac{1}{N} \sum_{i=1}^{N} CS_i^{wo,oracle}
$$

Required Experiment B report items:

- `Understanding Acc`
- `Reasoning Acc`
- `Chain Success`
- `Chain Success (w/o Track)`
- `Understanding Acc @ OracleTrack`
- `Reasoning Acc @ OracleTrack`
- `Chain Success (w/o Track)` in the OracleTrack rerun section

## 14. Experiment C: Protocol Comparison

Experiment C compares:

- `fixed-budget`,
- `model-native`.

The exact model subset is not fixed yet. It must be chosen after Experiment A from the strongest offline and streaming-native baselines.

### 14.1 Fixed-Budget Arm

Use the exact main protocol defined in this document.

### 14.2 Model-Native Arm

For streaming-native models:

- allow native full-history access over $[0, t_q]$,
- still forbid any future information,
- do not force fixed-budget sampling.

For offline models:

- allow the model’s default long-video input protocol,
- still forbid future information,
- the adapter is model-specific.

Important:

- model-native is an evaluation protocol comparison only,
- model-native results do not enter the main benchmark table.

## 15. Experiment D: Sampling / Context Ablation

Experiment D is one-factor-at-a-time, applies only to window-based tasks, and excludes `Spatial_Temporal_Grounding`.

Single-frame tasks do not participate in Experiment D because the ablation axes are temporal context length and temporal sampling density.

The exact model subset is not fixed yet. It must be chosen after Experiment A from the strongest offline and streaming-native baselines.

### 15.1 Window Ablation

Use:

- `16s @ 2fps -> B=32`
- `32s @ 2fps -> B=64`
- `64s @ 2fps -> B=128`

This ablation changes context span and allows budget to co-vary.

### 15.2 FPS Ablation

Use:

- `32s @ 1fps -> B=32`
- `32s @ 2fps -> B=64`
- `32s @ 4fps -> B=128`

This ablation changes sampling density and allows budget to co-vary.

### 15.3 No Separate Budget Axis

There is no standalone `budget-only` ablation. Budget variation is already induced by the `window` and `fps` axes.

## 16. Failure Handling and Implementation Safeguards

The evaluator should use the following deterministic failure behavior:

- if a required output field is missing, that component fails,
- if a bbox cannot be parsed, that bbox component fails,
- if a sampled interval cannot be parsed, the temporal component fails,
- if the judge output cannot be parsed as valid schema, judge pass is `0`,
- if tracking boxes are missing on required frames, those frames receive IoU $0$.

Recommended engineering behavior:

- log parse errors,
- retain raw model output for debugging,
- save normalized output alongside metrics,
- save sampled-to-original mappings for every sample.

## 17. Recommended Evaluation Artifacts

For each evaluated sample, store:

```json
{
  "sample_id": "...",
  "task_name": "...",
  "video_id": "...",
  "protocol": "fixed-budget",
  "sampled_frames_original": [10, 15, 20],
  "normalized_prediction": {},
  "component_metrics": {},
  "component_pass": {},
  "task_pass": 1
}
```

For each model and task, store:

```json
{
  "model_name": "...",
  "task_name": "...",
  "num_samples": 0,
  "task_accuracy": 0.0,
  "judge_pass_rate": 0.0,
  "bbox_pass_rate": 0.0,
  "tiou_pass_rate": 0.0,
  "tracking_mean_iou_mean": 0.0,
  "tracking_pass_rate_mean": 0.0,
  "mean_bertscore_f1": 0.0,
  "median_bertscore_f1": 0.0
}
```

For chain evaluation, store:

```json
{
  "experiment_b": {
    "base": {
      "num_chain_samples": 0,
      "num_scored_chain_samples": 0,
      "num_pending_chain_samples": 0,
      "understanding_acc": 0.0,
      "understanding_acc_by_task": {
        "Continuous_Actions_Caption": 0.0,
        "Spatial_Temporal_Grounding": 0.0
      },
      "understanding_acc_wo_track": 0.0,
      "understanding_acc_wo_track_by_task": {
        "Continuous_Actions_Caption": 0.0,
        "Spatial_Temporal_Grounding": 0.0
      },
      "reasoning_acc": 0.0,
      "chain_success": 0.0,
      "chain_success_wo_track": 0.0
    },
    "oracle_language": {
      "num_chain_samples": 0,
      "num_scored_chain_samples": 0,
      "num_pending_chain_samples": 0,
      "understanding_acc_wo_track": 0.0,
      "understanding_acc_wo_track_by_task": {
        "Continuous_Actions_Caption": 0.0,
        "Spatial_Temporal_Grounding": 0.0
      },
      "reasoning_acc": 0.0,
      "chain_success_wo_track": 0.0
    }
  }
}
```

## 18. Non-Negotiable Summary

The implementation must preserve the following decisions exactly:

- main protocol uses fixed budget $B=64$,
- single-frame tasks consume only one frame,
- short window tasks use backward nominal $2$ fps history sampling,
- long window tasks use uniform 64-frame sampling over the full question interval,
- STG uses GT answer interval expanded by $\pm 5s$ for input construction,
- all model temporal outputs are expressed in sampled-frame indices,
- evaluation converts sampled indices back to original-frame indices through an explicit mapping,
- `Commentary` is reported separately and excluded from main overall average,
- `Continuous_Actions_Caption`, `Continuous_Events_Caption`, and `Commentary` use LLM judge for joint text-plus-timing evaluation,
- `Temporal_Causal` passes if the main cause is correct and there is no key hallucination,
- `Objects_Spatial_Relationships` requires both GT object labels to be present and each matched box to satisfy IoU $\ge 0.5$,
- tracking uses mean IoU and frame pass rate, both thresholded at $0.5$,
- STG still evaluates time window separately with tIoU@0.5,
- OracleTrack is a true rerun with GT tracking substituted,
- Experiment D excludes STG and has no separate budget-only axis.
