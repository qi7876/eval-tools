# OmniChainBench Evaluation Framework

Language:

- English: `README.md`
- 中文: [README.zh-CN.md](/home/qi7876/dev/eval-tools/README.zh-CN.md)

This repository provides the evaluation framework for OmniChainBench. It has two goals:

1. Convert the raw benchmark dataset into reusable prepared test data.
2. Evaluate model outputs consistently across all benchmark tasks and experiments.

The framework is designed around a prepared-data workflow. You build the sampled model inputs once, cache them under a configured `prepared_root` such as `/data/public_data/mllmbenchmark_prepared`, and then reuse them across all baseline models. This avoids repeatedly decoding videos and rebuilding sampled frames for each run.

## What The Framework Covers

The current implementation includes:

- Raw dataset validation
- Chain manifest generation for Experiment B
- Prepared-data generation for the built-in `main` protocol and importable custom protocols
- Live adapter-based evaluation
- Framework-owned prompt building, chain-history injection, structured extraction, and scoring
- LLM-as-a-judge integration through an OpenAI-compatible API
- Experiment A summary
- Experiment B summary, including OracleTrack reruns

It does not yet ship the 10 baseline adapters themselves. You plug models in through the adapter interface described below.

## Project Layout

Key files:

- [pyproject.toml](/home/qi7876/dev/eval-tools/pyproject.toml): package metadata and dependencies
- [EXPERIMENT_EVALUATION_SPEC.md](/home/qi7876/dev/eval-tools/EXPERIMENT_EVALUATION_SPEC.md): benchmark specification
- [src/omnichain_eval/cli.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/cli.py): CLI entrypoint
- [src/omnichain_eval/config.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/config.py): TOML config loading
- [src/omnichain_eval/dataset.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/dataset.py): raw data loading and validation
- [src/omnichain_eval/protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/protocols.py): sampling rules
- [src/omnichain_eval/prepare.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prepare.py): prepared-data cache builder
- [src/omnichain_eval/prompting.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prompting.py): model-input prompt rendering and chain history building
- [src/omnichain_eval/normalize.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/normalize.py): strict structured-output validation
- [src/omnichain_eval/structurer.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/structurer.py): fixed structurer LLM integration
- [src/omnichain_eval/template_pack.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/template_pack.py): shared Markdown prompt-template loading
- [src/omnichain_eval/metrics.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/metrics.py): scoring logic
- [src/omnichain_eval/judge.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/judge.py): judge backend
- [src/omnichain_eval/experiments.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/experiments.py): experiment orchestration
- [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py): adapter interface
- `prompts/benchmark_v1/`: task-specific inference prompt pack
- `prompts/benchmark_oracle_v1/`: OracleTrack upstream inference prompt base directory with `language/`, `visual/`, and `language_visual/` variants
- `prompts/structurer_v1/`: task-specific structurer prompt pack
- `prompts/structurer_oracle_v1/`: OracleTrack upstream structurer prompt pack
- `prompts/judge_v1/`: task-specific judge prompt pack

Prompt-template convention:

- each Markdown file is the final prompt template body
- there are no `# system` / `# user` sections anymore
- benchmark, structurer, and judge all send user-only prompts at runtime
- `prompts/benchmark_v1/` and `prompts/structurer_v1/` each contain the 10 benchmark tasks
- `prompts/judge_v1/` contains the 9 judge-evaluated tasks; `Spatial_Temporal_Grounding` is rule-based and does not use judge prompts
- `prompts/benchmark_oracle_v1/` only covers OracleTrack upstream reruns for `Continuous_Actions_Caption` and `Spatial_Temporal_Grounding`, and is split into `language/`, `visual/`, and `language_visual/` subpacks
- `prompts/structurer_oracle_v1/` only covers OracleTrack upstream structuring for `Continuous_Actions_Caption` and `Spatial_Temporal_Grounding`
- `configs/examples/`: example TOML configs for common workflows

Generated directories:

- configured `prepared_root` such as `/data/public_data/mllmbenchmark_prepared/`: prepared sample bundles
- `artifacts/`: run-time predictions and summaries

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

Notes:

- `UV_CACHE_DIR=/tmp/uv-cache` is recommended in this environment because the default cache path may be unwritable.
- Video decoding uses PyAV only. If PyAV is unavailable, `prepare-data` fails loudly.

## Dataset Assumptions

The framework expects a raw dataset root configured through TOML. In the server examples used in this repository, that root is `/data/public_data/mllmbenchmark`.

At minimum, it assumes:

- main annotation files like `<data_root>/<sport>/<event>/<video_id>.json`
- videos alongside them as `<data_root>/<sport>/<event>/<video_id>.mp4`
- tracking files in `mot/*.txt`

The loader resolves tracking paths from the raw annotation fields, including paths written like `./data/...` or `./dataset/...`.

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
uv run omnichain-eval <command> --config <path/to/config.toml>
```

Available commands:

- `validate-data`
- `build-chain-manifest`
- `prepare-data`
- `run-eval`

All operational parameters are now managed through TOML files. Each command reads only the section it needs from the TOML file you pass in.

## Configuration Files

The framework is configuration-first.

That means:

- raw data paths live in TOML
- prepared-data paths live in TOML
- protocol specs live in TOML
- model adapter paths live in TOML
- inference prompt roots live in TOML
- prepare-data worker count lives in TOML
- structurer backend and prompt roots live in TOML
- judge backend and prompt roots live in TOML

This makes it practical to maintain one config per experiment, per protocol, or per model.

Example config files shipped with the repository:

- [configs/examples/workflow.toml](/home/qi7876/dev/eval-tools/configs/examples/workflow.toml): validate-data and build-chain-manifest
- [configs/examples/prepare_main.toml](/home/qi7876/dev/eval-tools/configs/examples/prepare_main.toml): prepare-data for `main`
- [configs/examples/prepare_custom_protocol.toml](/home/qi7876/dev/eval-tools/configs/examples/prepare_custom_protocol.toml): prepare-data for an importable custom protocol
- [configs/examples/run_eval_adapter.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_adapter.toml): mock smoke-test evaluation
- [configs/examples/run_eval_main.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_main.toml): live `run-eval` example for `main`
- [configs/examples/run_eval_custom_protocol.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_custom_protocol.toml): live `run-eval` example for an importable custom protocol

Supported top-level sections:

- `[validate_data]`
- `[build_chain_manifest]`
- `[prepare_data]`
- `[run_eval]`
- `[structurer]`
- `[judge]`

Minimal example:

```toml
[run_eval]
prepared_root = "/data/public_data/mllmbenchmark_prepared"
protocol = "main"
artifacts_root = "artifacts/runs"
prompt_root = "prompts/benchmark_v1"
adapter = "your_package.adapters.video:YourVideoAdapter"
chain_manifest = "artifacts/chain_pairs.jsonl"

[structurer]
backend = "openai"
prompt_root = "prompts/structurer_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2

[structurer.extra_body]
enable_thinking = false

[judge]
backend = "openai"
prompt_root = "prompts/judge_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2

[judge.extra_body]
enable_thinking = false
```

Path rules:

- relative paths are resolved relative to the TOML file location
- absolute paths are supported directly
- `protocol` / `protocols` entries accept either a built-in id such as `main` or an importable Python spec such as `your_package.protocols:EightFrameUniformProtocol`
- different experiments should use different TOML files
- keep `[run_eval]`, `[structurer]`, and `[judge]` together in the same run-eval TOML
- secrets can stay out of TOML by setting `api_key_env`

## Typical End-To-End Workflow

### 1. Validate the raw dataset

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  validate-data \
  --config configs/examples/workflow.toml
```

What it does:

- scans all main annotation JSON files
- resolves linked tracking files
- validates task schemas
- validates `upstream_annotation_id` for `Spatial_Imagination`

Behavior:

- exits `0` if supported tasks have no validation issues
- exits `1` if any supported-task issue is found
- prints up to 50 issues to the terminal

Important:

- unsupported raw tasks are reported separately and ignored by the main evaluation pipeline
- invalid supported annotations are reported individually
- a broken supported sample does not automatically make the whole file unusable during prepared-data generation

### 2. Build the Experiment B chain manifest

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --config configs/examples/workflow.toml
```

What it does:

- reads every supported `Spatial_Imagination` sample
- resolves its `upstream_annotation_id`
- validates that the upstream task is either `Continuous_Actions_Caption` or `Spatial_Temporal_Grounding`
- emits one JSONL row per chain pair

Important:

- unsupported tasks are ignored during scanning and do not block chain-manifest generation

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

Build the built-in `main` protocol cache:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --config configs/examples/prepare_main.toml
```

The shipped example config builds the reusable `main` cache under one prepared-data tree.
If a model needs its own native sampling rule, point `[prepare_data].protocols` at an importable protocol class instead.

What it does:

- loads raw samples
- applies the configured protocol sampling rule
- decodes the required frames
- writes each sample as a prepared bundle
- can additionally encode a sampled MP4 from the sampled frames when `[prepare_data].media_formats` includes `sampled_video`
- can additionally generate Oracle visual-overlay media when `[prepare_data].generate_oracle_visual_media = true`
- parallelizes work across videos when `[prepare_data].workers > 1`

Important:

- only supported benchmark tasks are prepared
- unsupported tasks are ignored and recorded in protocol metadata
- supported-task validation errors still fail the command before cache generation
- `workers` controls video-level thread concurrency inside one protocol; protocols are still built sequentially
- the shipped example prepare configs use `media_formats = ["frames", "sampled_video"]`
- `configs/examples/prepare_main.toml` also enables `generate_oracle_visual_media = true` for Experiment B Oracle visual variants
- `configs/examples/prepare_custom_protocol.toml` shows how to reference an importable native-sampling protocol
- sampled videos are rebuilt from the sampled frames only, use H.264 (`libx264`), contain no audio stream, and are generated only for multi-frame samples

The cache is sample-centric. The runtime evaluation path reads from the configured `prepared_root` instead of decoding raw videos again.

### 4. Evaluate a model

`run-eval` now uses live adapter mode only.

The runtime writes and resumes from six artifact files inside each run directory:

- `predictions.jsonl`: independent tasks and chain-upstream tasks
- `structured_predictions.jsonl`: structured outputs for those samples
- `results.jsonl`: completed evaluation records for those samples
- `chain_predictions.jsonl`: chain-downstream raw outputs
- `chain_structured_predictions.jsonl`: structured outputs for chain-downstream samples
- `chain_results.jsonl`: completed evaluation records for chain-downstream samples

After each run, the framework recomputes the latest aggregate metrics and rewrites `summary.json`.

## Prepared-Data Protocols

The framework now supports two kinds of protocol specs:

- built-in ids such as `main`
- importable Python classes such as `your_package.protocols:EightFrameUniformProtocol`

Notes:

- `main` remains the built-in benchmark protocol and supports all benchmark tasks, including STG
- custom protocols are how you express model-native sampling rules such as fixed 8-frame uniform sampling or 1 fps sampling
- a protocol owns sampling only; prompts, structuring, judging, metrics, chain logic, and OracleTrack remain framework-owned
- `prepare-data` and `run-eval` must use the same protocol spec; mismatches fail loudly and require rebuilding prepared data

### Custom Protocol Interface

Custom protocols must inherit from `BaseProtocol` in [src/omnichain_eval/protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/protocols.py).
The repository also ships [src/omnichain_eval/example_protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/example_protocols.py) as an importable example module for the example TOMLs.

Minimum shape:

```python
from omnichain_eval.protocols import BaseProtocol, uniform_sample_closed_interval


class EightFrameUniformProtocol(BaseProtocol):
    @property
    def protocol_id(self) -> str:
        return "native_uniform_8"

    @property
    def description(self) -> str:
        return "Eight-frame uniform sampling over the question window."

    def sample_frames(self, sample):
        if sample.timestamp_frame is not None and sample.task_name in {"Scoreboard_Single", "Objects_Spatial_Relationships"}:
            return [sample.timestamp_frame]
        if sample.q_window is None:
            raise ValueError(f"{sample.sample_id} is missing Q_window_frame")
        start, end = sample.q_window
        return uniform_sample_closed_interval(start, end, 8)
```

Then reference it from TOML:

```toml
[prepare_data]
protocols = ["omnichain_eval.example_protocols:ExampleEightFrameUniformProtocol"]

[run_eval]
protocol = "omnichain_eval.example_protocols:ExampleEightFrameUniformProtocol"
```

## Prepared Data Directory Layout

After running `prepare-data`, each resolved protocol gets its own directory under `prepared_root`:

```text
<prepared_root>/
  <protocol_id>/
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
          sampled_video.mp4
          oracle_visual/
            frames/
              0000.jpg
              0001.jpg
              ...
            sampled_video.mp4
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
- `question_text`
- `sampled_frames_original`
- `sampled_to_original`
- `frame_files`
- `sampled_video_file`
- `sampled_video_fps`
- `oracle_visual_frame_files`
- `oracle_visual_sampled_video_file`
- `reference_payload`
- `q_window` or `a_window`
- `upstream_annotation_id` when present
- `metadata`

`frame_files` and `sampled_video_file` are stored as bundle-relative paths in `manifest.json`.
At runtime, `load_prepared_samples()` rewrites them to absolute paths.
`sampled_video_file` is present only for multi-frame samples when sampled-video generation was enabled.
`sampled_video_fps` records the effective playback rate implied by the sampled frame spacing and is used both by adapters and by benchmark prompt rendering.
When Oracle visual media generation is enabled, Oracle upstream samples also include `oracle_visual_frame_files` and `oracle_visual_sampled_video_file`.
`metadata` also includes the resolved `protocol_spec` and `protocol_manifest` at runtime.

For segment tasks, `reference_payload` contains both original-frame and sampled-frame segments.

For tracking tasks:

- `Continuous_Actions_Caption` includes `tracking_gt_sampled`
- `Spatial_Temporal_Grounding` includes `time_window_sampled` and `tracking_gt_sampled`

### `build_manifest.json`

Protocol-level metadata:

- `protocol_id`
- `protocol_spec`
- `protocol_manifest`
- `data_status` with raw-dataset counts, supported-dataset counts, ignored unsupported task counts, and supported issue counts
- `supported_dataset_fingerprint`
- number of prepared samples

`stats.json` also includes `ignored_unsupported_sample_count` and `ignored_unsupported_task_counts`.

## Live Adapter Evaluation

### Adapter Resolution

`[run_eval].adapter` accepts either:

- `mock`
- `module.path:ClassName`

Examples:

```toml
[run_eval]
adapter = "mock"
prompt_root = "prompts/benchmark_v1"

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
```

```toml
[run_eval]
adapter = "my_project.adapters.qwen:QwenVideoAdapter"
prompt_root = "prompts/benchmark_v1"

[structurer]
backend = "openai"
prompt_root = "prompts/structurer_v1"
```

The adapter class must be importable in the current Python environment.
[run_eval].protocol must use the same built-in id or Python protocol spec that was used during `prepare-data`.
`[run_eval].prompt_root` is required and must point to a prompt pack directory containing the 10 task Markdown templates.
`[structurer].prompt_root` is also required and must point to the structurer prompt pack.
For real adapter bring-up, OracleTrack should stay enabled by default unless you are explicitly doing a non-Oracle-only milestone.
When `[run_eval].enable_oracle_track = true`, `[run_eval].oracle_prompt_root` and `[structurer].oracle_prompt_root` are also required.
`[run_eval].oracle_prompt_root` must point to the Oracle prompt base directory containing `language/`, `visual/`, and `language_visual/`.
`[structurer].oracle_prompt_root` must point to the Oracle upstream structurer prompt pack.

### Adapter Interface

Your adapter must inherit from `BaseModelAdapter` in [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py).

Required interface:

```python
from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import ModelInput


class MyAdapter(BaseModelAdapter):
    @property
    def name(self) -> str:
        return "my-model"

    def predict(
        self,
        model_input: ModelInput,
    ) -> str:
        ...
```

### What The Adapter Receives

The adapter now receives a `ModelInput` object. Important fields are:

- `model_input.messages`: final rendered prompt messages, already built by the framework
- `model_input.sample`: the full `PreparedSample`

Inside `model_input.sample`, common fields include:

- `sample_id`
- `protocol_id`
- `task_name`
- `question_text`
- `sampled_frames_original`
- `sampled_to_original`
- `frame_files`
- `sampled_video_file`
- `sampled_video_fps`
- `reference_payload`
- `q_window` or `a_window`
- `metadata`

In normal model adapters you should ignore `model_input.sample.reference_payload`, since it is GT. It is present because the framework also supports the built-in `mock` adapter and oracle-tracking workflows.

In practice, adapters read prompts from `model_input.messages` and choose media from `model_input.sample`:

- use `frame_files` for image-native models
- use `sampled_video_file` for video-native models when it is present
- use `sampled_video_fps` if the model API needs explicit timing metadata
- during Oracle visual reruns, consume the framework-provided Oracle overlay media the same way, without adapter-side Oracle branching

Important:

- the current protocol has already decided which frames or sampled video the model should see
- adapters should not re-sample or silently replace that sampling policy
- if a model needs a different native sampling rule, implement a new `BaseProtocol` class and rebuild prepared data for that protocol

For chain-downstream `Spatial_Imagination`, the framework automatically builds the final message list as:

- `user`: upstream question
- `assistant`: upstream answer
- `user`: current downstream prompt rendered from the benchmark template

### What The Adapter Should Return

The adapter should return the model's raw answer as a string.

The framework then:

- sends that raw answer to the fixed structurer module
- validates the structured JSON
- passes the validated structured output into scoring and judge logic

Canonical expectations by task:

- All bbox / tracking coordinates use the `normalized_1000` coordinate system.
  The top-left corner of the frame is `(0, 0)` and the bottom-right corner is `(1000, 1000)`.

- Text-only tasks:

```json
{"text": "..."}
```

- `Scoreboard_Single`:

```json
{"text": "...", "bbox": [x1, y1, x2, y2]}
```

- `Objects_Spatial_Relationships`:

```json
{
  "text": "...",
  "objects": [
    {"label": "Player A", "bbox": [x1, y1, x2, y2]},
    {"label": "Player B", "bbox": [x1, y1, x2, y2]}
  ]
}
```

During structuring and validation, required OSR labels are still enforced by label.
If a required label is missing entirely, the framework inserts that label in GT order with the sentinel bbox `[-1, -1, -1, -1]` so label-aligned IoU can still be computed.
If a label row is present, its bbox is not auto-repaired by the framework; schema-invalid rows still fail structuring.
The text answer is still judged normally, while a sentinel object's IoU becomes `0`.

- `Continuous_Events_Caption`:

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
- all bbox / tracking coordinates must stay in the `normalized_1000` coordinate system
- tracking boxes must use MOT format `[left, top, width, height]`
- malformed or missing fields fail deterministically

### Adapter Implementation Example

This example shows the minimum structure. The model call is pseudocode.

```python
from pathlib import Path

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import ModelInput


class MyVideoAdapter(BaseModelAdapter):
    @property
    def name(self) -> str:
        return "my-video-model"

    def predict(
        self,
        model_input: ModelInput,
    ) -> str:
        sample = model_input.sample
        messages = model_input.messages_as_dicts()
        image_paths = [Path(path) for path in sample.frame_files]
        video_path = Path(sample.sampled_video_file) if sample.sampled_video_file else None

        # Replace this block with your real model call.
        # Return the raw model answer as a string.
        if sample.task_name == "Scoreboard_Single":
            return '{"text": "The score is 1-0.", "bbox": [52, 833, 521, 907]}'

        if video_path is not None:
            _ = video_path
        else:
            _ = image_paths
        _ = messages
        return '{"text": "placeholder"}'
```

Run it like this:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_my_model.toml
```

The TOML should contain at least:

```toml
[run_eval]
prepared_root = "/data/public_data/mllmbenchmark_prepared"
protocol = "main"
artifacts_root = "artifacts/runs"
prompt_root = "prompts/benchmark_v1"
adapter = "my_package.my_adapter:MyVideoAdapter"
chain_manifest = "artifacts/chain_pairs.jsonl"
enable_oracle_track = true
oracle_prompt_root = "prompts/benchmark_oracle_v1"

[judge]
backend = "openai"
prompt_root = "prompts/judge_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2
concurrency = 1

[judge.extra_body]
enable_thinking = false

[structurer]
backend = "openai"
prompt_root = "prompts/structurer_v1"
oracle_prompt_root = "prompts/structurer_oracle_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2
concurrency = 1

[structurer.extra_body]
enable_thinking = false
```

This assumes prepared data was built with Oracle visual media enabled, for example via `configs/examples/prepare_main.toml`.

## Structurer Configuration

Structurer settings live in the `[structurer]` section of the TOML file used by `run-eval`.
`[structurer].prompt_root` must point to a directory containing one Markdown template per task.
Each Markdown file is the final user prompt body sent to the structurer model.

Typical OpenAI-compatible configuration:

```toml
[structurer]
backend = "openai"
prompt_root = "prompts/structurer_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2
concurrency = 1

[structurer.extra_body]
enable_thinking = false
```

`[structurer].extra_body` is passed through directly to the OpenAI-compatible request body.
`[structurer].temperature` is configurable; the default is `0`.
If you do not override them, the framework defaults to `model = "qwen3.5-397b-a17b"`, `temperature = 0`, and `extra_body.enable_thinking = false`.

For local smoke tests you can skip the external structurer API and use:

```toml
[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
invalid_json_retries = 1
concurrency = 1
```

Current structurer behavior:

- prompts are task-specific rather than one shared generic prompt
- the structurer may do light normalization of explicit values from the raw model output
- it uses the task-specific schema and prompt template to determine which canonical fields the current task expects
- it should prefer the final answer if the raw output contains reasoning plus a final answer
- it must not invent missing boxes, intervals, tracking rows, or answer text that do not appear in the raw model output
- for `Scoreboard_Single`, the framework accepts an explicit sentinel bbox `[-1, -1, -1, -1]` from the structurer, but does not auto-fill missing bbox fields
- for `Objects_Spatial_Relationships`, the framework reorders objects by GT label and auto-fills missing required labels with the sentinel bbox `[-1, -1, -1, -1]`, but does not auto-repair bbox fields for labels that already appear

Retry behavior:

- if the structurer response is not valid JSON, the framework retries
- if the structurer response is JSON but fails the strict task schema, the framework also retries
- retry count is controlled by `[structurer].invalid_json_retries`
- if all attempts still fail, that sample is deferred to the next run without writing a structured artifact row
- structurer execution runs asynchronously in the background; `[structurer].concurrency` controls the number of concurrent workers

## Judge Configuration

Judge settings live in the `[judge]` section of the TOML file used by `run-eval`.
`[judge].prompt_root` must point to a directory containing one Markdown template per judge-evaluated task.
Each Markdown file is the final user-only prompt body sent to the judge model.

Typical OpenAI-compatible configuration:

```toml
[judge]
backend = "openai"
prompt_root = "prompts/judge_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2
concurrency = 1

[judge.extra_body]
enable_thinking = false
```

`[judge].extra_body` is also forwarded unchanged into the request body.
`[judge].temperature` is configurable; the default is `0`.
If you do not override them, the framework defaults to `model = "qwen3.5-397b-a17b"`, `temperature = 0`, and `extra_body.enable_thinking = false`.

Then export only the secret:

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

You can use `[judge].api_key` directly in TOML, but `api_key_env` is usually cleaner. When both judge and structurer use DashScope, they can share the same `DASHSCOPE_API_KEY`.

For local smoke tests you can avoid external judge calls by changing the backend:

```toml
[judge]
backend = "static-pass"
prompt_root = "prompts/judge_v1"
```

or:

```toml
[judge]
backend = "static-fail"
prompt_root = "prompts/judge_v1"
```

Retry behavior:

- if the judge response is not valid JSON, the framework retries
- if the judge response is valid JSON but has missing keys, wrong field names, empty required fields, or non-binary score fields, the framework also retries
- retry count is controlled by `[judge].invalid_json_retries`
- if all attempts still return malformed judge responses, that sample is deferred to the next run; its prediction artifact remains, but no completed result row is written this round
- judge execution runs asynchronously in the background; `[judge].concurrency` controls the number of concurrent evaluation workers

## Resume Evaluation

`run-eval` now supports resumable execution at sample granularity.

Behavior:

- predictions for independent tasks and chain-upstream tasks are appended to `predictions.jsonl`
- structured outputs for those samples are appended to `structured_predictions.jsonl`
- completed evaluation records for those samples are appended to `results.jsonl`
- chain-downstream predictions are appended to `chain_predictions.jsonl`
- structured outputs for chain-downstream samples are appended to `chain_structured_predictions.jsonl`
- completed chain-downstream evaluation records are appended to `chain_results.jsonl`
- if the process is interrupted, rerunning the same config will skip completed samples
- if interruption happened after prediction was written but before structuring finished, the runner reuses the saved prediction and only reruns structuring
- if interruption happened after structuring was written but before scoring finished, the runner reuses the saved structured output and only reruns scoring
- if a chain-upstream answer is missing, the corresponding chain-downstream sample remains blocked until the next run
- resumability is derived from those six artifact files, not from a separate failure list

Important:

- resumability depends on writing into the same run directory
- in practice you should set a fixed `[run_eval].run_name` for long-running jobs
- OracleTrack pair evaluation is resumed independently through:
  `oracle_language_pair_results.jsonl`,
  `oracle_visual_pair_results.jsonl`,
  and `oracle_language_visual_pair_results.jsonl`

## Running Experiment A

Example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_main.toml
```

Outputs are written under:

```text
artifacts/runs/<timestamp>_<model_name>_<protocol_id>/
```

Files:

- `run.log`
- `predictions.jsonl`
- `structured_predictions.jsonl`
- `results.jsonl`
- `chain_predictions.jsonl`
- `chain_structured_predictions.jsonl`
- `chain_results.jsonl`
- `summary.json`

Artifact semantics:

- `run.log` stores the detailed runtime log for this invocation, including stage transitions, aggregated prediction/structurer/judge progress, OracleTrack variant progress, and structurer/judge failure payloads
- `predictions.jsonl` stores raw model outputs for independent tasks and chain-upstream tasks
- `structured_predictions.jsonl` stores validated structured outputs for those samples
- `results.jsonl` stores completed evaluation records for those samples
- `chain_predictions.jsonl` stores raw model outputs for chain-downstream samples
- `chain_structured_predictions.jsonl` stores validated structured outputs for chain-downstream samples
- `chain_results.jsonl` stores completed evaluation records for chain-downstream samples
- if a sample already has a prediction but structuring did not complete, it remains absent from its corresponding `*_structured_predictions.jsonl` file and will be retried on the next run
- if a sample already has a structured output but judge or scoring did not complete, it remains absent from its corresponding `*_results.jsonl` file and will be retried on the next run

`summary.json` includes:

- `data_status` copied from the protocol `build_manifest.json`
- total target sample count for this run
- completed counts before this invocation, in this invocation, and in total
- `pending_prediction_sample_ids`
- `predicted_not_structured_sample_ids`
- `structured_not_evaluated_sample_ids`
- `overall`
- per-task summaries
- `experiment_b` if a chain manifest was provided
- `pending_chain_prediction_sample_ids`
- `chain_predicted_not_structured_sample_ids`
- `chain_structured_not_evaluated_sample_ids`
- `blocked_chain_sample_ids`
- error summaries for normal, chain, and oracle evaluation in this invocation

Unsupported-task information lives in `data_status`. Unsupported samples are not counted as runtime pending items and are not included in task accuracies.

## Running Experiment B

Experiment B requires a chain manifest:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --config configs/examples/workflow.toml
```

Then set `[run_eval].chain_manifest` in the TOML used by `run-eval`:

```toml
[run_eval]
chain_manifest = "artifacts/chain_pairs.jsonl"
```

The current runner writes `experiment_b` as grouped summaries:

- `base`: understanding accuracy, reasoning accuracy, chain success, chain success (w/o track)
- `oracle_language`: text-only Oracle rerun with language GT tracking injection
- `oracle_visual`: text-only Oracle rerun with visual GT tracking injection
- `oracle_language_visual`: text-only Oracle rerun with both injections

Each Oracle group reports:

- `num_chain_samples`
- `num_scored_chain_samples`
- `num_pending_chain_samples`
- `understanding_acc`
- `reasoning_acc`
- `chain_success_wo_track`

## OracleTrack

OracleTrack is framework-owned.
For real adapter implementation and handoff, OracleTrack compatibility is the default target rather than an optional follow-up.

Then run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_oracle.toml
```

with:

```toml
[run_eval]
enable_oracle_track = true
chain_manifest = "artifacts/chain_pairs.jsonl"
```

What the framework does:

- reruns the upstream and downstream pair through the adapter
- uses `[run_eval].oracle_prompt_root` as a 3-variant Oracle prompt base directory and `[structurer].oracle_prompt_root` as the dedicated Oracle upstream structurer prompt pack
- runs three Oracle variants: `language`, `visual`, and `language_visual`
- in `language`, injects GT tracking directly into the upstream Oracle prompt body in `normalized_1000` coordinates
- in `visual`, switches the upstream media to prepared Oracle visual-overlay frames/video with GT tracking boxes drawn on them
- in `language_visual`, applies both injections together
- tells the model that the subject has already been identified by GT tracking, so the Oracle upstream output should omit tracking boxes
- rebuilds downstream chain history from the full rendered upstream prompt plus the upstream raw answer
- scores Oracle upstream samples only on the non-tracking component, then reports Oracle text-only chain metrics in Experiment B

## BERTScore

BERTScore is always computed during evaluation when a task provides textual comparison inputs.

Notes:

- BERTScore is supplementary only
- it never changes pass/fail
- the framework clamps the BERTScore tokenizer max length to the underlying model's `max_position_embeddings` when the tokenizer reports an oversized sentinel length

## Failure Behavior

The framework follows deterministic failure rules:

- missing required output field -> fail that component
- malformed bbox -> fail spatial component
- malformed sampled interval -> fail temporal component
- malformed judge response -> retry up to `[judge].invalid_json_retries`, then leave the sample unfinished for the next run without writing a completed sample result
- missing tracking prediction on required sampled frame -> IoU `0`

Additionally:

- duplicate tracking predictions on the same sampled frame are collapsed by keeping the first one
- explicit sentinel boxes in `Scoreboard_Single`, and explicit or missing-label auto-filled sentinel boxes in `Objects_Spatial_Relationships`, contribute IoU `0`
- unsupported raw tasks are ignored and recorded in `build_manifest.json` / `summary.json`
- supported-task validation problems still stop `prepare-data`

## Running Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

The test suite currently covers:

- protocol sampling
- chain manifest generation
- prepared-data building
- mock adapter evaluation
- structurer retry and validation behavior
- resumable prediction -> structuring -> evaluation flow
- Experiment B summary flow

## Common Recipes

### Smoke test everything with the built-in mock adapter

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval build-chain-manifest --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --config configs/examples/prepare_main.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval run-eval --config configs/examples/run_eval_adapter.toml
```

## Current Limitations

- The repository does not yet include concrete adapters for the 10 baseline models.
- The repository does not yet include a built-in library of baseline-specific native sampling protocol classes.
- Prepared-data generation currently writes JPEG frame bundles per sample rather than a deduplicated shared frame store.

## Recommended Workflow For Adding A New Model

1. Create a dedicated TOML file for the model and protocol you want to evaluate.
2. If the model needs native sampling different from `main`, implement a `BaseProtocol` subclass in your own importable module.
3. Run `prepare-data` for the protocol(s) you need.
4. Implement a `BaseModelAdapter` subclass in your own importable module.
5. Point `[run_eval].prompt_root` to your inference prompt pack and `[structurer].prompt_root` to your structurer prompt pack.
6. Make the adapter consume `model_input.sample`, `model_input.messages`, and the prepared media referenced by the sample.
7. Return the raw model answer as a string.
8. Run `run-eval --config your_model.toml` on the same protocol you prepared.
9. Set `[run_eval].chain_manifest` to get Experiment B metrics.
10. Keep `[run_eval].enable_oracle_track = true` by default and let the framework run OracleTrack reruns, unless you were explicitly asked to deliver a non-Oracle-only milestone.

If you follow that flow, the model integration stays thin: all dataset parsing, frame preparation, prompt construction, chain accounting, structured extraction, scoring, and summary generation remain inside the framework.

In practice the adapter only owns two things:

- how to turn `model_input.sample.frame_files` or `model_input.sample.sampled_video_file` plus `model_input.messages` into a real model call
- how to return the model's raw answer as a string
