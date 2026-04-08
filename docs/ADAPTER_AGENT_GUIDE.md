# Adapter Implementation Guide For A Coding Agent

This document is the implementation handoff for wiring a concrete video model into the current OmniChainBench evaluation framework.

It is written against the current code in this repository, not against earlier designs.

## Goal

Implement one importable adapter class that lets:

```bash
uv run omnichain-eval run-eval --config <run_eval.toml>
```

call the model through the framework and produce the normal runtime artifacts:

- `predictions.jsonl`
- `structured_predictions.jsonl`
- `results.jsonl`
- `chain_predictions.jsonl`
- `chain_structured_predictions.jsonl`
- `chain_results.jsonl`
- `summary.json`

The adapter must be minimal. The framework already owns prompt construction, chain-history construction, structuring, judging, scoring, artifact writing, and resume.

Unless the repository owner explicitly says otherwise, OracleTrack compatibility is part of the default completion target for a real adapter handoff.
That does not mean adding Oracle-specific branches inside the adapter.
It means the adapter must work unchanged when the framework runs OracleTrack reruns through the normal `ModelInput` contract.

## Read These Files First

- [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py)
- [src/omnichain_eval/schema.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/schema.py)
- [src/omnichain_eval/prompting.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prompting.py)
- [src/omnichain_eval/cli.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/cli.py)
- [src/omnichain_eval/experiments.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/experiments.py)
- [configs/examples/run_eval_adapter.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_adapter.toml)
- [configs/examples/run_eval_main.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_main.toml)
- [configs/examples/run_eval_custom_protocol.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_custom_protocol.toml)
- [README.md](/home/qi7876/dev/eval-tools/README.md)

For live evaluation examples, the repository ships configs for both the built-in `main` protocol and a placeholder custom-protocol workflow under `configs/examples/`.

## Current Framework Contract

### Adapter interface

The only required interface is:

```python
class BaseModelAdapter(ABC):
    def predict(self, model_input: ModelInput) -> str: ...
```

### What `ModelInput` contains

The adapter receives [ModelInput](/home/qi7876/dev/eval-tools/src/omnichain_eval/schema.py#L132):

- `model_input.sample`: full [PreparedSample](/home/qi7876/dev/eval-tools/src/omnichain_eval/schema.py#L62)
- `model_input.messages`: ordered chat-style prompt messages

Useful `PreparedSample` fields:

- `sample.sample_id`
- `sample.task_name`
- `sample.question_text`
- `sample.frame_files`
- `sample.sampled_video_file`
- `sample.sampled_video_fps`
- `sample.sampled_frames_original`
- `sample.sampled_to_original`
- `sample.metadata["bundle_dir"]`

Important detail:

- `load_prepared_samples()` rewrites `sample.frame_files` to absolute paths at runtime in [src/omnichain_eval/prepare.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prepare.py#L329).
- `load_prepared_samples()` also rewrites `sample.sampled_video_file` to an absolute path when it exists.
- The adapter can open frame files or the sampled video directly from those absolute paths.
- The current protocol has already decided which sampled frames or sampled video the model should see.
- If a model needs a different native sampling rule, implement a `BaseProtocol` class and rebuild prepared data for that protocol instead of re-sampling inside the adapter.
- During OracleTrack visual reruns, the framework may swap `sample.frame_files` and `sample.sampled_video_file` to prepared Oracle visual-overlay media for the upstream sample.
- The adapter must consume that swapped media transparently, without any Oracle-specific branch.

### What the framework already does

Do not re-implement any of this inside the adapter:

- task-specific prompt template rendering
- output-format instruction injection
- multi-turn chain history construction
- raw-output structuring
- llm-as-a-judge scoring
- metric computation
- artifact persistence
- resume from existing `*.jsonl` files

The framework constructs the input and the adapter only feeds it into the model.

### Prompt packs the framework owns

The adapter does not choose prompts. The framework loads them from TOML:

- `prompts/benchmark_v1/`: 10 benchmark tasks
- `prompts/judge_v1/`: 9 judge-evaluated tasks
- `prompts/benchmark_oracle_v1/`: OracleTrack upstream prompt base directory with `language/`, `visual`, and `language_visual/` variants for `Continuous_Actions_Caption` and `Spatial_Temporal_Grounding`
- `prompts/structurer_oracle_v1/`: OracleTrack upstream structurer prompt pack for those same two tasks

## Message Semantics You Must Preserve

### Non-chain samples

For normal tasks, `model_input.messages` contains one final user message:

- role: `user`
- content: the fully rendered benchmark prompt

Use this rendered content, not `sample.question_text`, as the actual final prompt sent to the model.

### Chain downstream samples

For `Spatial_Imagination` downstream evaluation, the framework automatically builds history in [build_chain_history()](/home/qi7876/dev/eval-tools/src/omnichain_eval/prompting.py#L142).

The adapter will receive messages in this exact order:

1. upstream rendered user prompt
2. upstream assistant raw model output
3. downstream rendered user prompt

This is verified in [tests/test_run_eval_resume.py](/home/qi7876/dev/eval-tools/tests/test_run_eval_resume.py#L338).

Do not drop or rewrite this history. If the model API accepts chat messages, pass them through in order. If the model only accepts one text prompt, flatten the messages while preserving roles and order.

## Responsibilities Split

### The adapter owns

- loading the model
- caching the loaded model inside the adapter instance
- turning `frame_files` or `sampled_video_file` plus `messages` into the model's native inference input
- calling generation
- returning the raw model output string

### The adapter does not own

- JSON validation
- bbox/tracking parsing
- any fallback post-processing with another LLM
- prompt-template selection
- chain-history assembly
- dataset scanning
- sampling-policy definition
- any report/summary logic

## Recommended Implementation Shape

Preferred location if you want the adapter inside this repo:

- `src/omnichain_eval/adapters/<model_name>.py`

Alternative:

- any importable module path in the same Python environment, as long as TOML can reference it via `module.path:ClassName`

Keep the adapter file thin. Put model-specific helper functions near it only if needed.

## Required Behavior

### 1. Lazy model loading

Load heavy model state lazily, for example in `__init__()` or in a dedicated `_ensure_loaded()` method called from `predict()`.

Reason:

- `resolve_adapter()` instantiates the class once
- `run-eval` reuses that adapter instance across the whole run
- loading per sample would be unusably slow

### 2. Raw output only

`predict()` must return the raw model answer string.

Do not:

- parse it into JSON
- normalize field names
- inject empty placeholders
- call the structurer yourself
- call the judge yourself

That work now belongs entirely to the framework.

### 3. Use the full rendered prompt

The last user message already includes task-specific instructions and the required output contract. The adapter should treat `model_input.messages` as the source of truth.

### 4. Preserve chain history

For downstream chain samples, forward the full history into the model. The adapter must not collapse the request to just the downstream question.

### 5. OracleTrack compatibility is required, but still framework-owned

OracleTrack is framework-owned. The framework runs three Oracle variants, injects GT tracking into the upstream rerun prompt when needed, swaps in Oracle visual-overlay media when needed, and rebuilds downstream history from the full rendered upstream prompt plus the upstream raw answer.

The adapter still does not need any OracleTrack-specific branch.
It should always just consume `sample.frame_files` / `sample.sampled_video_file` and `model_input.messages`.

However, OracleTrack-enabled `run-eval` is the default handoff target for a real adapter implementation.
Do not treat OracleTrack as an optional later step unless the repository owner explicitly asks for a non-Oracle-only milestone.

## Strong Recommendations

### Prefer in-process integration

Preferred path:

- import the model's Python entrypoint
- load weights in-process
- call inference directly

Do not introduce a separate offline mode. This repository intentionally removed offline evaluation mode.

If the model truly cannot run in the same environment, stop and report the exact blocker instead of silently designing a second evaluation path.

### Flatten messages explicitly if needed

Many video models do not support chat messages directly. In that case, add one explicit formatter, for example:

```python
def _flatten_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"].strip().upper()
        content = message["content"].strip()
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)
```

Use one deterministic format and keep it simple.

### Fail loudly on true integration errors

The framework already tolerates per-sample runtime failures and resume. The adapter should therefore raise real exceptions for:

- missing weights
- missing processor/tokenizer
- unreadable frame paths
- unsupported message format
- failed model generation

Do not hide these by returning fabricated text.

## Implementation Checklist

### Step 1. Find the model's narrowest reusable inference entrypoint

Before writing the adapter, identify:

- how the model accepts frames
- whether it expects frame paths, PIL images, tensors, or video files
- whether it accepts chat messages or a single text prompt
- what generation function returns

You want the smallest direct path from:

- `list[str]` absolute frame paths
- optional absolute sampled-video path
- `list[PromptMessage]` conversation

to:

- one generated text string

### Step 2. Decide adapter file placement

Recommended:

- add one adapter module under [src/omnichain_eval/adapters](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters)

If the model code already lives elsewhere in the same environment, you can place a thin wrapper there instead.

### Step 3. Implement the class

Skeleton:

```python
from __future__ import annotations

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import ModelInput


class YourModelAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "your-model-name"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # import model code here if import is heavy
        # load weights / processor here
        self._loaded = True

    def predict(self, model_input: ModelInput) -> str:
        self._ensure_loaded()
        sample = model_input.sample
        messages = model_input.messages_as_dicts()
        frame_paths = sample.frame_files
        video_path = sample.sampled_video_file

        # choose video_path or frame_paths based on model capability
        # convert messages + media to model-native input
        # run generation
        raw_text = ...

        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        return raw_text
```

### Step 4. Map framework input to model input

Typical mapping:

- `sample.frame_files` -> image/frame loader for the model
- `sample.sampled_video_file` -> video-native loader for the model
- `sample.sampled_video_fps` -> explicit timing metadata if the model API needs it
- `model_input.messages_as_dicts()` -> chat input or flattened text prompt

OracleTrack implication:

- if the framework swaps the upstream sample to Oracle visual-overlay media, the adapter should keep working without code changes
- if the framework injects GT tracking into the rendered Oracle prompt, the adapter should still just consume `model_input.messages`

Do not read GT from:

- `sample.reference_payload`

That field exists for internal mock/oracle workflows and must not be used by a real adapter.

### Step 5. Add a runnable TOML example for the real adapter

You will need a config similar to:

```toml
[run_eval]
prepared_root = "/data/public_data/mllmbenchmark_prepared"
protocol = "main"
artifacts_root = "../../artifacts/runs"
prompt_root = "../../prompts/benchmark_v1"
run_name = "your-model-main"
model_name = "your-model"
adapter = "omnichain_eval.adapters.your_model:YourModelAdapter"
chain_manifest = "../../artifacts/chain_pairs.jsonl"
enable_oracle_track = true
oracle_prompt_root = "../../prompts/benchmark_oracle_v1"

[structurer]
backend = "openai"
prompt_root = "../../prompts/structurer_v1"
oracle_prompt_root = "../../prompts/structurer_oracle_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2

[structurer.extra_body]
enable_thinking = false

[judge]
backend = "openai"
prompt_root = "../../prompts/judge_v1"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2

[judge.extra_body]
enable_thinking = false
```

If the model needs native sampling different from `main`, replace `protocol = "main"` with the same custom protocol spec you used during `prepare-data`, for example `your_package.protocols:YourNativeSamplingProtocol`.
The prepared data used by this config must have been built with Oracle visual media enabled, for example via `configs/examples/prepare_main.toml` or an equivalent custom prepare-data config with `generate_oracle_visual_media = true`.

The framework owns structurer and judge configuration. Do not try to push model runtime knobs into this TOML unless the repository explicitly decides to support that later.

### Step 6. Validate with a small run first

Recommended execution order:

1. `validate-data`
2. `build-chain-manifest`
3. `prepare-data` with Oracle visual media enabled
4. `run-eval` with OracleTrack enabled on a smoke subset if you have one, otherwise a normal small run

At minimum, confirm:

- the adapter imports correctly from TOML
- the model loads once
- the model receives absolute frame paths
- chain downstream calls receive 3 messages in order
- `predictions.jsonl` is populated
- `structured_predictions.jsonl` and `results.jsonl` continue updating after predictions
- Oracle upstream reruns also execute through the same adapter class without adapter-side branching
- Oracle visual upstream reruns work when the framework swaps in prepared overlay media
- rerunning the same `run_name` resumes instead of recomputing finished samples

## Resume Behavior You Must Understand

`run-eval` is resumable by artifact files in the run directory.

Normal channel:

- `predictions.jsonl`
- `structured_predictions.jsonl`
- `results.jsonl`

Chain downstream channel:

- `chain_predictions.jsonl`
- `chain_structured_predictions.jsonl`
- `chain_results.jsonl`

The framework loads existing rows, backfills state, and skips already completed samples in [src/omnichain_eval/cli.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/cli.py#L488).

Implication for the adapter:

- returning raw text consistently is enough
- you do not need any custom checkpointing in the adapter

## Testing Expectations

If you can add tests without needing the real model weights, do it.

Good test targets:

- message flattening helper
- frame-path collection helper
- lazy-load behavior
- preservation of message order when flattening multi-turn history
- preserving the framework-provided message and media inputs for OracleTrack reruns if you can unit-test that without real weights

If real-model inference is too heavy for CI, do not force it into CI. Keep unit tests local to pure helpers and do manual smoke validation with the actual model.

## Common Mistakes To Avoid

- Using `sample.question_text` as the whole prompt and ignoring rendered `messages`
- Ignoring chain history for downstream `Spatial_Imagination`
- Reading `reference_payload` in a real adapter
- Returning parsed dicts instead of raw strings
- Doing extra JSON cleanup inside the adapter
- Silently swallowing model failures and returning placeholder text
- Adding a second offline evaluation path
- Rebuilding any judge or structurer logic inside the adapter

## Acceptance Criteria

The adapter is complete when all of the following are true:

1. `adapter = "module.path:ClassName"` resolves successfully.
2. `run-eval` can call `predict()` on prepared samples without adapter-side prompt construction.
3. Normal samples write to `predictions.jsonl` and continue through structuring and judging.
4. Chain downstream samples write to `chain_predictions.jsonl` and preserve upstream rendered prompt plus upstream raw answer in history.
5. OracleTrack-enabled `run-eval` works with the same adapter class and without adapter-side Oracle branches.
6. Oracle upstream reruns consume the framework-provided prompt and media inputs unchanged, including Oracle visual-overlay media when present.
7. Re-running the same `run_name` resumes from existing artifacts.
8. The adapter returns only raw model text and leaves structuring/judging to the framework.

## If You Hit An Environment Blocker

Report the blocker in concrete terms:

- exact import that fails
- exact package or CUDA mismatch
- exact model entrypoint that cannot be reused

Do not solve that by inventing a parallel offline pipeline. The framework should stay on the single live-adapter path unless the repository owner explicitly changes direction.
