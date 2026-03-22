# OmniChainBench Evaluation Framework

This repository provides the evaluation framework for OmniChainBench. It has two goals:

1. Convert the raw benchmark dataset into reusable prepared test data.
2. Evaluate model outputs consistently across all benchmark tasks and experiments.

The framework is designed around a prepared-data workflow. You build the sampled model inputs once, cache them under `prepared_data/`, and then reuse them across all baseline models. This avoids repeatedly decoding videos and rebuilding sampled frames for each run.

## What The Framework Covers

The current implementation includes:

- Raw dataset validation
- Chain manifest generation for Experiment B
- Prepared-data generation for the main fixed-budget protocol and Experiment D fixed-budget ablations
- Live adapter-based evaluation
- Offline prediction replay
- Task normalization and scoring
- LLM-as-a-judge integration through an OpenAI-compatible API
- Experiment A summary
- Experiment B summary, including OracleTrack plumbing
- Consolidated report generation

It does not yet ship the 10 baseline adapters themselves. You plug models in through the adapter interface described below.

## Project Layout

Key files:

- [pyproject.toml](/home/qi7876/dev/eval-tools/pyproject.toml): package metadata and dependencies
- [EXPERIMENT_EVALUATION_SPEC.md](/home/qi7876/dev/eval-tools/EXPERIMENT_EVALUATION_SPEC.md): benchmark specification
- [src/omnichain_eval/cli.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/cli.py): CLI entrypoint
- [src/omnichain_eval/dataset.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/dataset.py): raw data loading and validation
- [src/omnichain_eval/protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/protocols.py): sampling rules
- [src/omnichain_eval/prepare.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prepare.py): prepared-data cache builder
- [src/omnichain_eval/normalize.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/normalize.py): prediction normalization
- [src/omnichain_eval/metrics.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/metrics.py): scoring logic
- [src/omnichain_eval/judge.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/judge.py): judge backend
- [src/omnichain_eval/experiments.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/experiments.py): experiment orchestration
- [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py): adapter interface

Generated directories:

- `prepared_data/`: prepared sample bundles
- `artifacts/`: predictions, per-sample results, summaries, reports, judge cache

## Installation

The project uses `uv`.

Basic setup:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

For development and tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

If you want supplementary BERTScore support:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra bertscore
```

Notes:

- `UV_CACHE_DIR=/tmp/uv-cache` is recommended in this environment because the default cache path may be unwritable.
- Video decoding uses PyAV if available, otherwise OpenCV fallback.

## Dataset Assumptions

The framework expects a `data/` root with benchmark annotations and videos. At minimum, it assumes:

- main annotation files like `data/<sport>/<event>/<video_id>.json`
- videos alongside them as `data/<sport>/<event>/<video_id>.mp4`
- commentary sidecar files like `commentary_<id>.json`
- tracking files in `mot/*.txt`

The loader resolves commentary and tracking paths from the raw annotation fields, including paths written like `./data/...` or `./dataset/...`.

Each annotation becomes a stable sample id:

```text
<sport>/<event>/<video_id>#<annotation_id>
```

Example:

```text
3x3_Basketball/Men/1#4
```

## Supported CLI Commands

The package installs a single CLI:

```bash
uv run omnichain-eval <command> ...
```

Available commands:

- `validate-data`
- `build-chain-manifest`
- `prepare-data`
- `run-eval`
- `report`

## Typical End-To-End Workflow

### 1. Validate the raw dataset

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --data-root data
```

What it does:

- scans all main annotation JSON files
- resolves linked commentary and tracking files
- validates task schemas
- validates `upstream_annotation_id` for `Spatial_Imagination`

Behavior:

- exits `0` if there are no issues
- exits `1` if any issue is found
- prints up to 50 issues to the terminal

Important:

- invalid annotations are reported individually
- a broken sample does not automatically make the whole file unusable during prepared-data generation

### 2. Build the Experiment B chain manifest

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --data-root data \
  --out artifacts/chain_pairs.jsonl
```

What it does:

- reads every `Spatial_Imagination` sample
- resolves its `upstream_annotation_id`
- validates that the upstream task is either `Continuous_Actions_Caption` or `Spatial_Temporal_Grounding`
- emits one JSONL row per chain pair

Output schema:

```json
{
  "pair_id": "video#downstream|video#upstream",
  "video_key": "Sport/Event/1",
  "upstream_sample_id": "Sport/Event/1#2",
  "downstream_sample_id": "Sport/Event/1#8",
  "upstream_task_name": "Continuous_Actions_Caption"
}
```

### 3. Prepare reusable test data

Main protocol only:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --data-root data \
  --prepared-root prepared_data \
  --protocol main
```

Main protocol plus all Experiment D fixed-budget ablations:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --data-root data \
  --prepared-root prepared_data \
  --protocol main \
  --protocol expd_window_16s_2fps \
  --protocol expd_window_32s_2fps \
  --protocol expd_window_64s_2fps \
  --protocol expd_fps_32s_1fps \
  --protocol expd_fps_32s_2fps \
  --protocol expd_fps_32s_4fps
```

What it does:

- loads raw samples
- applies the protocol-specific sampling rule
- decodes the required frames
- writes each sample as a prepared bundle

The cache is sample-centric. The runtime evaluation path reads from `prepared_data/` instead of decoding raw videos again.

### 4. Evaluate a model

There are two supported evaluation modes:

- live adapter mode
- offline prediction replay

Both modes use the same normalization, metrics, judge, and reporting logic after the prediction boundary.

### 5. Aggregate run summaries

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  report \
  --artifacts-root artifacts/runs \
  --out artifacts/report.json
```

This scans all run directories under `artifacts/runs` and writes a consolidated `report.json`.

## Prepared-Data Protocol IDs

Currently supported:

- `main`
- `expd_window_16s_2fps`
- `expd_window_32s_2fps`
- `expd_window_64s_2fps`
- `expd_fps_32s_1fps`
- `expd_fps_32s_2fps`
- `expd_fps_32s_4fps`

Notes:

- `main` supports all benchmark tasks, including STG
- Experiment D protocols exclude STG by design
- Experiment C `model-native` is not prebuilt in this version

## Prepared Data Directory Layout

After running `prepare-data --protocol main`, the layout looks like:

```text
prepared_data/
  main/
    build_manifest.json
    index.jsonl
    stats.json
    samples/
      <sport>/<event>/<video_id>/
        <annotation_id>/
          manifest.json
          frames/
            0000.jpg
            0001.jpg
            ...
```

### `index.jsonl`

One row per prepared sample:

```json
{
  "sample_id": "Sport/Event/1#4",
  "task_name": "Continuous_Actions_Caption",
  "video_key": "Sport/Event/1",
  "annotation_id": "4",
  "manifest_path": "samples/Sport/Event/1/4/manifest.json",
  "frame_count": 31
}
```

### `manifest.json`

Each bundle stores a serialized `PreparedSample`. Important fields include:

- `sample_id`
- `task_name`
- `protocol_id`
- `prompt_text`
- `sampled_frames_original`
- `sampled_to_original`
- `frame_files`
- `reference_payload`
- `q_window` or `a_window`
- `upstream_annotation_id` when present
- `metadata`

For segment tasks, `reference_payload` contains both original-frame and sampled-frame segments.

For tracking tasks:

- `Continuous_Actions_Caption` includes `tracking_gt_sampled`
- `Spatial_Temporal_Grounding` includes `time_window_sampled` and `tracking_gt_sampled`

### `build_manifest.json`

Protocol-level metadata:

- protocol spec
- dataset summary
- dataset fingerprint
- number of prepared samples
- up to 100 dataset issues skipped during preparation

## Live Adapter Evaluation

### Adapter Resolution

`run-eval --adapter ...` accepts either:

- `mock`
- `module.path:ClassName`

Examples:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter mock \
  --judge-backend static-pass
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_project.adapters.qwen:QwenVideoAdapter
```

The adapter class must be importable in the current Python environment.

### Adapter Interface

Your adapter must inherit from `BaseModelAdapter` in [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py).

Required interface:

```python
from typing import Any

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import PreparedSample


class MyAdapter(BaseModelAdapter):
    @property
    def name(self) -> str:
        return "my-model"

    def supports_commentary(self) -> bool:
        return True

    def supports_oracle_track(self) -> bool:
        return False

    def predict(
        self,
        sample: PreparedSample,
        *,
        oracle_track: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Any:
        ...
```

### What The Adapter Receives

The `PreparedSample` object includes:

- `sample_id`: stable benchmark sample identifier
- `task_name`: benchmark task name
- `protocol_id`: sampling protocol used
- `prompt_text`: question or query text
- `frame_files`: absolute paths to prepared JPEG frames
- `sampled_frames_original`: original frame indices for the prepared frames
- `sampled_to_original`: sampled index -> original frame mapping
- `reference_payload`: GT payload
- `q_window` or `a_window`
- `metadata`: misc bundle metadata including bundle directory

In normal model adapters you should ignore `reference_payload`, since it is GT. It is present because the framework also supports the built-in `mock` adapter and oracle-tracking workflows.

### What The Adapter Should Return

The adapter can return:

- a Python `dict`
- a JSON string
- a plain text string for text-only tasks

The evaluator will normalize the output into the benchmark’s canonical formats.

Canonical expectations by task:

- Text-only tasks:

```json
{"text": "..."}
```

- `Scoreboard_Single`:

```json
{"text": "...", "bbox": [xtl, ytl, xbr, ybr]}
```

- `Objects_Spatial_Relationships`:

```json
{"text": "...", "bbox_a": [xtl, ytl, xbr, ybr], "bbox_b": [xtl, ytl, xbr, ybr]}
```

- `Continuous_Events_Caption` and `Commentary`:

```json
{
  "segments": [
    {"start_sampled": 0, "end_sampled": 3, "text": "..."}
  ]
}
```

- `Continuous_Actions_Caption`:

```json
{
  "segments": [
    {"start_sampled": 0, "end_sampled": 3, "text": "..."}
  ],
  "tracking": [
    {"frame_sampled": 0, "bbox_mot": [left, top, width, height]}
  ]
}
```

- `Spatial_Temporal_Grounding`:

```json
{
  "time_window_sampled": [0, 4],
  "tracking": [
    {"frame_sampled": 0, "bbox_mot": [left, top, width, height]}
  ]
}
```

Important:

- all temporal outputs must be in sampled-frame indices, not original video frame indices
- tracking boxes must use MOT format `[left, top, width, height]`
- malformed or missing fields fail deterministically

### Adapter Implementation Example

This example shows the minimum structure. The model call is pseudocode.

```python
from pathlib import Path
from typing import Any

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import PreparedSample


class MyVideoAdapter(BaseModelAdapter):
    @property
    def name(self) -> str:
        return "my-video-model"

    def supports_commentary(self) -> bool:
        return False

    def predict(
        self,
        sample: PreparedSample,
        *,
        oracle_track: bool = False,
        context: dict[str, Any] | None = None,
    ) -> Any:
        image_paths = [Path(path) for path in sample.frame_files]
        prompt = sample.prompt_text

        # Replace this block with your real model call.
        # The result may be a dict or a JSON string.
        if sample.task_name == "Scoreboard_Single":
            return {
                "text": "The score is 1-0.",
                "bbox": [100, 900, 1000, 980],
            }

        return {"text": "placeholder"}
```

Run it like this:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_package.my_adapter:MyVideoAdapter \
  --judge-backend openai
```

## Offline Prediction Replay

If you have already run inference elsewhere, you can score the results without writing an adapter.

Input file format: JSONL

Each row must contain:

- `sample_id`
- either `raw_output` or `normalized_prediction`
- optional `protocol_id`

Example:

```json
{"sample_id": "Sport/Event/1#1", "raw_output": {"text": "The score is 1-0.", "bbox": [10, 20, 200, 80]}}
{"sample_id": "Sport/Event/1#4", "raw_output": {"text": "The athlete moves from left to right."}}
{"sample_id": "Sport/Event/1#5", "raw_output": {"segments": [{"start_sampled": 0, "end_sampled": 2, "text": "The athlete jogs into the frame."}]}}
```

Evaluate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --predictions artifacts/my_predictions.jsonl \
  --model-name my-offline-model
```

Notes:

- if `normalized_prediction` is present, the runner prefers it over `raw_output`
- normalization is still applied during scoring
- if a sample is missing from the file, it is evaluated as missing output and will fail deterministically

## Judge Configuration

By default, `run-eval` uses the OpenAI-compatible judge backend.

Required environment variables:

```bash
export EVAL_JUDGE_BASE_URL="http://your-judge-endpoint/v1"
export EVAL_JUDGE_API_KEY="your-api-key"
```

Optional:

```bash
export EVAL_JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"
```

For local smoke tests you can avoid external judge calls:

```bash
--judge-backend static-pass
--judge-backend static-fail
```

Judge responses are cached under:

```text
<artifacts-root>/judge_cache/
```

## Running Experiment A

Example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_package.my_adapter:MyVideoAdapter \
  --chain-manifest artifacts/chain_pairs.jsonl
```

Outputs are written under:

```text
artifacts/runs/<timestamp>_<model_name>_<protocol_id>/
```

Files:

- `predictions.jsonl`
- `sample_results.jsonl`
- `task_summaries.json`
- `summary.json`

`summary.json` includes:

- `overall`
- per-task summaries
- `experiment_b` if a chain manifest was provided
- whether commentary was supported

`Commentary` is reported separately and excluded from `overall`.

## Running Experiment B

Experiment B requires a chain manifest:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --data-root data \
  --out artifacts/chain_pairs.jsonl
```

Then include it in `run-eval`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_package.my_adapter:MyVideoAdapter \
  --chain-manifest artifacts/chain_pairs.jsonl
```

The current runner computes:

- understanding accuracy
- reasoning accuracy
- chain success

If oracle rerun information is available, it also computes:

- `understanding_acc_oracle`
- `reasoning_acc_oracle`
- `chain_success_oracle`

## OracleTrack

There are two supported OracleTrack paths.

### Live adapter mode

Your adapter must implement:

```python
def supports_oracle_track(self) -> bool:
    return True
```

Then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_package.my_adapter:MyVideoAdapter \
  --chain-manifest artifacts/chain_pairs.jsonl \
  --enable-oracle-track
```

What the framework does:

- reruns the upstream and downstream pair through the adapter
- sets `oracle_track=True` in the adapter call
- also passes `context={"chain_pair": ..., "role": "upstream" | "downstream"}`
- replaces tracking with GT during upstream scoring where the task requires it

### Offline mode

If you do not use a live adapter, provide a second prediction file:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --predictions artifacts/base_predictions.jsonl \
  --oracle-predictions artifacts/oracle_predictions.jsonl \
  --chain-manifest artifacts/chain_pairs.jsonl
```

The framework does not synthesize oracle reruns from the base predictions. You must provide explicit oracle outputs.

## Commentary Support

If a model does not support `Commentary`, there are two paths:

- in live adapter mode, return `False` from `supports_commentary()`
- in offline mode, pass `--commentary-unsupported`

Effect:

- commentary is marked `N/A`
- commentary is excluded from `overall`

## BERTScore

Enable it only if you installed the `bertscore` extra:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --adapter my_package.my_adapter:MyVideoAdapter \
  --enable-bertscore
```

Notes:

- BERTScore is supplementary only
- it never changes pass/fail

## Failure Behavior

The framework follows deterministic failure rules:

- missing required output field -> fail that component
- malformed bbox -> fail spatial component
- malformed sampled interval -> fail temporal component
- invalid judge JSON -> judge fail
- missing tracking prediction on required sampled frame -> IoU `0`

Additionally:

- duplicate tracking predictions on the same sampled frame are collapsed by keeping the first one
- invalid raw annotations are skipped during prepared-data build and recorded in `build_manifest.json`

## Running Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
```

The test suite currently covers:

- protocol sampling
- chain manifest generation
- prepared-data building
- mock adapter evaluation
- Experiment B summary flow

## Common Recipes

### Smoke test everything with the built-in mock adapter

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval build-chain-manifest --data-root data --out artifacts/chain_pairs.jsonl
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --data-root data --prepared-root prepared_data --protocol main
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval run-eval --prepared-root prepared_data --protocol main --adapter mock --judge-backend static-pass --chain-manifest artifacts/chain_pairs.jsonl
```

### Score an offline JSONL export

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --prepared-root prepared_data \
  --protocol main \
  --predictions artifacts/my_model_predictions.jsonl \
  --model-name my-model \
  --chain-manifest artifacts/chain_pairs.jsonl
```

### Build all fixed-budget caches in advance

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --data-root data \
  --prepared-root prepared_data \
  --protocol main \
  --protocol expd_window_16s_2fps \
  --protocol expd_window_32s_2fps \
  --protocol expd_window_64s_2fps \
  --protocol expd_fps_32s_1fps \
  --protocol expd_fps_32s_2fps \
  --protocol expd_fps_32s_4fps
```

## Current Limitations

- The repository does not yet include concrete adapters for the 10 baseline models.
- Experiment C model-native inputs are not standardized in this version.
- Prepared-data generation currently writes JPEG frame bundles per sample rather than a deduplicated shared frame store.
- Judge retry logic is not implemented; invalid judge JSON fails directly.

## Recommended Workflow For Adding A New Model

1. Run `prepare-data` for the protocol(s) you need.
2. Implement a `BaseModelAdapter` subclass in your own importable module.
3. Make the adapter consume `sample.frame_files` and `sample.prompt_text`.
4. Return task outputs in the benchmark’s sampled-frame coordinate space.
5. Run `run-eval --adapter ...` on `main`.
6. Add `--chain-manifest` to get Experiment B metrics.
7. If needed, implement `supports_oracle_track()` and handle `oracle_track=True`.
8. When the model is stable, run the Experiment D protocol ids from the same prepared cache root.

If you follow that flow, the model integration stays thin: all dataset parsing, frame preparation, normalization, scoring, reporting, and chain accounting remain inside the framework.
