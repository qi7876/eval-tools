# OmniChainBench 测评框架

这是 OmniChainBench 的评测与数据准备工具。它的核心目标有两个：

1. 把原始 benchmark 数据集转换成可复用的 prepared test data。
2. 以统一、可复现的方式对不同模型进行评测与汇总。

这个框架的设计重点是“先预构建、后复用”。

也就是说，你先对原始视频做一次采样和解码，生成 `prepared_data/` 缓存；后续评测 10 个 baseline 模型时，直接复用这些缓存，不再重复从视频解码和采样。这样可以显著降低重复开销，也能保证不同模型看到的是完全一致的输入。

## 当前能力

当前版本已经实现：

- 原始数据校验
- Experiment B 所需 chain manifest 生成
- 主协议 `main` 和 Experiment D 固定预算协议的 prepared-data 构建
- 在线 adapter 模式评测
- 各任务输出归一化
- 指标计算与样本级 pass/fail 判断
- 通过 OpenAI-compatible API 的 LLM-as-a-judge
- Experiment A 汇总
- Experiment B 汇总，包括 OracleTrack 流程接口

当前还没有内置 10 个 baseline 模型的 adapter。你需要按下面的 adapter 接口把模型接进来。

## 仓库结构

核心文件：

- [pyproject.toml](/home/qi7876/dev/eval-tools/pyproject.toml)：Python 项目与依赖配置
- [EXPERIMENT_EVALUATION_SPEC.md](/home/qi7876/dev/eval-tools/EXPERIMENT_EVALUATION_SPEC.md)：实验规范
- [src/omnichain_eval/cli.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/cli.py)：CLI 入口
- [src/omnichain_eval/config.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/config.py)：TOML 配置加载
- [src/omnichain_eval/dataset.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/dataset.py)：原始数据加载和校验
- [src/omnichain_eval/protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/protocols.py)：采样协议
- [src/omnichain_eval/prepare.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prepare.py)：prepared-data 构建
- [src/omnichain_eval/normalize.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/normalize.py)：模型输出归一化
- [src/omnichain_eval/metrics.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/metrics.py)：指标计算
- [src/omnichain_eval/judge.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/judge.py)：Judge 接口
- [src/omnichain_eval/experiments.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/experiments.py)：实验编排
- [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py)：模型 adapter 接口
- `configs/examples/`：常见流程的 TOML 示例配置

运行后产生的目录：

- `prepared_data/`：预构建后的测试数据缓存
- `artifacts/`：运行期预测结果和 summary

## 安装方式

本项目使用 `uv`。

基础安装：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

安装开发依赖：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

如果需要开启 BERTScore：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra bertscore
```

说明：

- 在当前环境下建议显式设置 `UV_CACHE_DIR=/tmp/uv-cache`，避免默认缓存目录不可写。
- 视频解码优先使用 PyAV；如果环境里没有 PyAV，则自动回退到 OpenCV。

## 数据集假设

框架默认原始数据根目录为 `data/`，目录结构至少满足：

- 主标注：`data/<sport>/<event>/<video_id>.json`
- 视频：`data/<sport>/<event>/<video_id>.mp4`
- Commentary GT：`commentary_<id>.json`
- Tracking GT：`mot/*.txt`

每个 annotation 会被转换成稳定的 `sample_id`：

```text
<sport>/<event>/<video_id>#<annotation_id>
```

例如：

```text
3x3_Basketball/Men/1#4
```

框架会自动解析原始标注中的：

- `question` / `query`
- `Q_window_frame`
- `A_window_frame`
- `commentary`
- `tracking_bboxes`
- `upstream_annotation_id`

并把它们整理成评测所需的标准内部结构。

## CLI 命令一览

统一入口：

```bash
uv run omnichain-eval <command> --config <path/to/config.toml>
```

已实现命令：

- `validate-data`
- `build-chain-manifest`
- `prepare-data`
- `run-eval`

现在所有运行参数都通过 TOML 管理。命令本身只接收 `--config`，然后从对应的 TOML section 里读取参数。

## TOML 配置方式

当前框架采用 config-first 的方式。

也就是说，下面这些内容都应该放在 TOML 里：

- 原始数据路径
- prepared-data 路径
- protocol id
- adapter 路径
- judge backend 与生成参数

这样你就可以为不同实验、不同协议、不同模型分别维护不同的 TOML 文件。

仓库内提供了这些示例配置：

- [configs/examples/workflow.toml](/home/qi7876/dev/eval-tools/configs/examples/workflow.toml)：数据校验、chain manifest、prepare-data
- [configs/examples/run_eval_adapter.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_adapter.toml)：在线 adapter 评测
- [configs/examples/run_eval_expd_window_32s_2fps.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_expd_window_32s_2fps.toml)：Experiment D 独立配置示例

支持的顶层 section：

- `[validate_data]`
- `[build_chain_manifest]`
- `[prepare_data]`
- `[run_eval]`
- `[judge]`

最小 `run-eval` 示例：

```toml
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
artifacts_root = "artifacts/runs"
adapter = "your_package.adapters.video:YourVideoAdapter"
chain_manifest = "artifacts/chain_pairs.jsonl"

[judge]
backend = "openai"
base_url = "http://your-judge-endpoint/v1"
api_key_env = "EVAL_JUDGE_API_KEY"
model = "gpt-4.1-mini"
invalid_json_retries = 2
```

路径规则：

- 相对路径都以当前 TOML 文件所在目录为基准解析
- 不同实验建议拆成不同 TOML 文件
- 密钥更建议通过 `[judge].api_key_env` 指向环境变量

## 标准工作流

推荐的完整流程如下：

1. 校验原始数据
2. 生成 Experiment B 的 chain manifest
3. 预构建 prepared data
4. 用 adapter 进行评测

下面按步骤说明。

## 第一步：校验原始数据

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  validate-data \
  --config configs/examples/workflow.toml
```

这个命令会：

- 扫描所有主标注 JSON
- 解析 linked commentary 文件
- 解析 MOT tracking 文件
- 校验任务 schema
- 校验 `Spatial_Imagination` 是否带有合法的 `upstream_annotation_id`

返回行为：

- 没有问题时退出码为 `0`
- 有问题时退出码为 `1`
- 终端打印前 50 个问题

重要说明：

- 当前实现按 annotation 粒度容错
- 某个 sample 损坏时，只会跳过该 sample，不会整份 JSON 全部失效

## 第二步：生成 Experiment B 的 chain manifest

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --config configs/examples/workflow.toml
```

这个命令会：

- 找到所有 `Spatial_Imagination`
- 读取其 `upstream_annotation_id`
- 在同一个源 annotation 文件内定位上游 sample
- 校验上游任务是否是：
  - `Continuous_Actions_Caption`
  - `Spatial_Temporal_Grounding`
- 生成 Experiment B 所需的显式 pair 列表

输出 JSONL 示例：

```json
{
  "pair_id": "Sport/Event/1#8|Sport/Event/1#2",
  "video_key": "Sport/Event/1",
  "upstream_sample_id": "Sport/Event/1#2",
  "downstream_sample_id": "Sport/Event/1#8",
  "upstream_task_name": "Continuous_Actions_Caption"
}
```

## 第三步：预构建测试数据

这是整个框架最关键的一步。

你应该先构建 prepared data，然后后续所有 baseline 模型都只复用 prepared data，不再重复从原视频采样。

### 只构建主协议

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --config configs/examples/workflow.toml
```

上面的示例配置已经在 `[prepare_data].protocols` 中放入了 `main` 和当前所有 Experiment D 固定预算协议。

如果你只想给单个实验准备缓存，建议单独新建一个 TOML，例如：

```toml
[prepare_data]
data_root = "data"
prepared_root = "prepared_data"
protocols = ["main"]
```

这个命令会：

- 按协议对 sample 进行采样
- 解码相应视频帧
- 把每个 sample 写成一个独立 bundle
- 存储 sampled-to-original 映射
- 存储任务评测需要的 GT sidecar

### 为什么要先构建 prepared data

因为 benchmark 中多个模型会共享完全相同的采样输入。若每次评测模型时都重新从原视频读取和采样，会有几个问题：

- 重复解码视频，开销大
- 不同模型运行时容易出现输入不一致
- 不便于复现实验
- OracleTrack / chain rerun 等流程会更难统一

prepared-data 方案解决了这些问题。

## 支持的协议 ID

当前可用的协议：

- `main`
- `expd_window_16s_2fps`
- `expd_window_32s_2fps`
- `expd_window_64s_2fps`
- `expd_fps_32s_1fps`
- `expd_fps_32s_2fps`
- `expd_fps_32s_4fps`

说明：

- `main` 支持全部任务，包括 STG
- Experiment D 协议按规范不包含 STG
- Experiment C 的 `model-native` 当前没有统一 prepared-data 形式

## Prepared Data 的目录结构

执行 `prepare-data`，并且对应配置的 `[prepare_data].protocols` 包含 `main` 后，目录类似：

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

每个 sample 一行：

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

每个 bundle 对应一个 `PreparedSample`，关键字段包括：

- `sample_id`
- `task_name`
- `protocol_id`
- `question_text`
- `sampled_frames_original`
- `sampled_to_original`
- `frame_files`
- `reference_payload`
- `q_window` 或 `a_window`
- `upstream_annotation_id`
- `metadata`

### `reference_payload` 的作用

它保存了评测所需的 GT 标准形式。

例如：

- Text 任务会保存 `text`
- Segment 任务会保存：
  - `segments_original`
  - `segments_sampled`
- `Continuous_Actions_Caption` 会多出：
  - `tracking_original`
  - `tracking_gt_sampled`
- `Spatial_Temporal_Grounding` 会多出：
  - `time_window_original`
  - `time_window_sampled`
  - `tracking_gt_sampled`

### `build_manifest.json`

协议级元数据，包括：

- protocol 定义
- dataset summary
- dataset fingerprint
- prepared sample 数量
- 本次构建使用的已校验数据元信息

## 第四步：运行评测

当前 `run-eval` 只保留在线 adapter 模式。

每个 run 目录下会维护 4 个可续测文件：

- `predictions.jsonl`：独立任务和链式上游任务的原始输出
- `results.jsonl`：这些样本的完成态评测结果
- `chain_predictions.jsonl`：链式下游任务的原始输出
- `chain_results.jsonl`：链式下游任务的完成态评测结果

每次 run 结束后，框架都会基于当前最新结果重写 `summary.json`。

## 在线 Adapter 模式

### adapter 参数的写法

`[run_eval].adapter` 支持两种形式：

- `mock`
- `module.path:ClassName`

例如：

```toml
[run_eval]
adapter = "mock"
prompt_root = "prompts/benchmark_v1"
```

或者：

```toml
[run_eval]
adapter = "my_project.adapters.qwen:QwenVideoAdapter"
prompt_root = "prompts/benchmark_v1"
```

`[run_eval].prompt_root` 现在是必填项，必须指向一个包含 11 个任务 Markdown 模板的 prompt 目录。

### adapter 接口

你需要继承 [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py) 中的 `BaseModelAdapter`。

接口如下：

```python
from typing import Any

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import ModelInput


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
        model_input: ModelInput,
    ) -> Any:
        ...
```

### adapter 会收到什么

adapter 现在会收到一个 `ModelInput`。常用字段包括：

- `model_input.task_name`
- `model_input.frame_files`
- `model_input.messages`
- `model_input.oracle_track`
- `model_input.sample`

其中 `model_input.sample` 里常见字段有：

- `sample_id`
- `protocol_id`
- `question_text`
- `sampled_frames_original`
- `sampled_to_original`
- `reference_payload`
- `q_window`
- `a_window`
- `metadata`

通常你直接把 `model_input.frame_files` 和 `model_input.messages` 送给模型即可。

说明：

- `frame_files` 是绝对路径
- 每张图已经按采样顺序排好
- sampled 索引就是 `frame_files` 的顺序索引

如果当前 sample 是链式下游 `Spatial_Imagination`，框架会自动把最终消息构造成：

- `system`
- `user`：上游问题
- `assistant`：上游答案
- `user`：当前下游问题

### adapter 应返回什么

可以返回：

- Python `dict`
- JSON 字符串
- 对于纯文本任务，也可以直接返回普通字符串

评测器会自动调用 normalization，把输出转成 benchmark 需要的标准结构。

### 各任务标准输出格式

#### Text-only 任务

```json
{"text": "..."}
```

#### `Scoreboard_Single`

```json
{"text": "...", "bbox": [xtl, ytl, xbr, ybr]}
```

#### `Objects_Spatial_Relationships`

```json
{
  "text": "...",
  "bbox_a": [xtl, ytl, xbr, ybr],
  "bbox_b": [xtl, ytl, xbr, ybr]
}
```

#### `Continuous_Events_Caption` / `Commentary`

```json
{
  "segments": [
    {"start_sampled": 0, "end_sampled": 3, "text": "..."}
  ]
}
```

#### `Continuous_Actions_Caption`

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

#### `Spatial_Temporal_Grounding`

```json
{
  "time_window_sampled": [0, 4],
  "tracking": [
    {"frame_sampled": 0, "bbox_mot": [left, top, width, height]}
  ]
}
```

重要约束：

- 所有时间输出都必须使用 sampled-frame index
- 不允许输出 original frame index 作为预测时间坐标
- tracking box 使用 MOT 格式 `[left, top, width, height]`
- 缺字段或格式错误会按规范直接失败

### 一个最小 adapter 示例

```python
from pathlib import Path
from typing import Any

from omnichain_eval.adapters.base import BaseModelAdapter
from omnichain_eval.schema import ModelInput


class MyVideoAdapter(BaseModelAdapter):
    @property
    def name(self) -> str:
        return "my-video-model"

    def supports_commentary(self) -> bool:
        return False

    def predict(
        self,
        model_input: ModelInput,
    ) -> Any:
        image_paths = [Path(path) for path in model_input.frame_files]
        messages = model_input.messages_as_dicts()
        sample = model_input.sample

        # 在这里替换成你的真实模型调用逻辑
        if sample.task_name == "Scoreboard_Single":
            return {
                "text": "The score is 1-0.",
                "bbox": [100, 900, 1000, 980]
            }

        return {"text": "placeholder"}
```

然后这样运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_my_model.toml
```

对应 TOML 至少需要：

```toml
[run_eval]
prepared_root = "prepared_data"
protocol = "main"
artifacts_root = "artifacts/runs"
prompt_root = "prompts/benchmark_v1"
adapter = "my_package.my_adapter:MyVideoAdapter"

[judge]
backend = "openai"
base_url = "http://your-judge-endpoint/v1"
api_key_env = "EVAL_JUDGE_API_KEY"
model = "gpt-4.1-mini"
invalid_json_retries = 2
concurrency = 1
```

## Judge 配置

Judge 的配置写在 `run-eval` 所使用 TOML 的 `[judge]` section 里。

典型配置如下：

```toml
[judge]
backend = "openai"
base_url = "http://your-judge-endpoint/v1"
api_key_env = "EVAL_JUDGE_API_KEY"
model = "gpt-4.1-mini"
temperature = 0.0
top_p = 1.0
top_k = 1
max_tokens = 256
n = 1
seed = 42
invalid_json_retries = 2
concurrency = 1
```

然后只在环境变量里提供密钥：

```bash
export EVAL_JUDGE_API_KEY="your-api-key"
```

当然，你也可以把 `api_key` 直接写进 TOML，但更推荐用 `api_key_env`。

如果只是本地 smoke test，不想真的调 judge API，可以把 backend 改成：

```toml
[judge]
backend = "static-pass"
```

或者：

```toml
[judge]
backend = "static-fail"
```

当前 retry 规则：

- judge 返回内容如果不是合法 JSON，会自动重试
- judge 返回内容即使是合法 JSON，只要字段名不对、缺关键字段、关键字段为空、或 0/1 评分字段异常，也会自动重试
- 重试次数由 `[judge].invalid_json_retries` 控制
- 如果所有尝试返回的 judge 响应都仍然不合规，则该样本会保留已有 prediction，但本轮不会写入对应的 `*_results.jsonl`，留到下一轮继续
- Judge 在后台异步执行；`[judge].concurrency` 控制并发 worker 数

## 断点续测

`run-eval` 现在支持按 sample 级别断点续测。

行为如下：

- 独立任务和链式上游任务的推理结果写入 `predictions.jsonl`
- 这些样本的完成态评测写入 `results.jsonl`
- 链式下游任务的推理结果写入 `chain_predictions.jsonl`
- 链式下游任务的完成态评测写入 `chain_results.jsonl`
- 如果中途中断，再次使用同一个配置运行时，会自动跳过已完成样本
- 如果恰好中断在“预测已写入、评分未完成”，则会复用已保存的预测，只补做评分
- 如果链式上游样本还没有产出回答，对应下游样本会保持 blocked，等下一轮继续
- 续测状态不是靠单独的失败清单，而是每轮都重新比对这 4 个 artifact 文件

重要说明：

- 断点续测依赖于写入同一个 run 目录
- 实际使用时，建议为长任务显式设置固定的 `[run_eval].run_name`
- OracleTrack 的 pair 级评测也会通过 `oracle_pair_results.jsonl` 断点恢复

## 运行 Experiment A

主 benchmark 的典型运行方式：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_main.toml
```

产物会写到：

```text
artifacts/runs/<timestamp>_<model_name>_<protocol_id>/
```

其中包括：

- `predictions.jsonl`
- `results.jsonl`
- `chain_predictions.jsonl`
- `chain_results.jsonl`
- `summary.json`

这些文件的语义是：

- `predictions.jsonl`：独立任务和链式上游任务的原始模型输出
- `results.jsonl`：这些样本的完成态评测结果
- `chain_predictions.jsonl`：链式下游任务的原始模型输出
- `chain_results.jsonl`：链式下游任务的完成态评测结果
- 如果某个 sample 已经完成预测，但 judge 或评分流程没有完成，那么它不会写入对应的 `*_results.jsonl`，下一轮会继续补测

`summary.json` 会额外记录：

- 本轮目标 sample 总数
- 本轮开始前已完成数、本轮新完成数、累计完成数
- `pending_prediction_sample_ids`
- `predicted_not_evaluated_sample_ids`
- `overall`
- 各 task 的 summary
- `experiment_b`
- 是否支持 commentary
- `pending_chain_prediction_sample_ids`
- `chain_predicted_not_evaluated_sample_ids`
- `blocked_chain_sample_ids`
- 本轮 normal / chain / oracle 失败摘要

说明：

- `Commentary` 单独报告
- `overall` 不包含 `Commentary`

## 运行 Experiment B

Experiment B 依赖 chain manifest：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --config configs/examples/workflow.toml
```

然后在 `run-eval` 的 TOML 中设置 `[run_eval].chain_manifest`：

```toml
[run_eval]
chain_manifest = "artifacts/chain_pairs.jsonl"
```

当前会计算：

- understanding accuracy
- reasoning accuracy
- chain success

如果提供了 OracleTrack rerun，还会额外计算：

- `understanding_acc_oracle`
- `reasoning_acc_oracle`
- `chain_success_oracle`

## OracleTrack

有两种路径。

### 在线 adapter 模式

如果你的 adapter 支持 OracleTrack，需要实现：

```python
def supports_oracle_track(self) -> bool:
    return True
```

然后运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  run-eval \
  --config path/to/run_eval_oracle.toml
```

对应配置：

```toml
[run_eval]
enable_oracle_track = true
chain_manifest = "artifacts/chain_pairs.jsonl"
```

此时框架会：

- 对 chain pair 做 rerun
- 调用 adapter 时传入 `oracle_track=True`
- 同时通过 `context` 提供 chain metadata
- 在上游评分时用 GT tracking 替换 tracking 分量

## Commentary 支持

如果模型不支持 commentary，直接让 adapter 返回 `supports_commentary() == False`。

效果：

- commentary 记为 `N/A`
- `overall` 不受影响

## BERTScore

需要先安装 `bertscore` extra：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --extra bertscore
```

然后运行时开启：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval run-eval --config path/to/run_eval_with_bertscore.toml
```

对应配置：

```toml
[run_eval]
enable_bertscore = true
```

说明：

- BERTScore 仅做补充指标
- 不影响 pass/fail

## 失败处理规则

框架遵循确定性失败规则：

- 缺少必要字段 -> 该分量失败
- bbox 解析失败 -> 空间分量失败
- sampled interval 解析失败 -> 时间分量失败
- judge 响应格式异常 -> 最多重试 `[judge].invalid_json_retries` 次，之后该样本留待下一轮继续，且不会写入完成态的 sample result
- tracking 缺失 -> 对应 frame 的 IoU 记为 `0`

此外：

- 同一 sampled frame 上多个 tracking 预测会保留第一个
- 原始数据中不合法的 annotation 会被跳过，并记录到 `build_manifest.json`

## 运行测试

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest
```

当前测试覆盖：

- 协议采样
- chain manifest 生成
- prepared-data 构建
- mock adapter 评测
- Experiment B 汇总流程

## 常用命令模板

### 全流程 smoke test

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval build-chain-manifest --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval run-eval --config configs/examples/run_eval_adapter.toml
```

### 预构建全部固定预算缓存

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --config configs/examples/workflow.toml
```

## 当前限制

- 还没有内置 10 个 baseline 的具体 adapter
- Experiment C 的 model-native 输入协议还没有统一
- prepared-data 当前按 sample 写 JPEG bundle，没有做共享帧去重存储

## 推荐接入新模型的方式

建议按以下顺序接入一个新模型：

1. 先为这个模型和协议新建一个独立 TOML 文件
2. 运行 `prepare-data`
3. 编写一个 `BaseModelAdapter` 子类
4. 在 TOML 里配置 `[run_eval].prompt_root`，并让 adapter 只消费：
   - `model_input.frame_files`
   - `model_input.messages`
5. 让模型返回 benchmark 要求的 canonical output
6. 先跑 `run-eval --config your_model.toml` 对 `main` 协议评测
7. 在 TOML 中设置 `[run_eval].chain_manifest` 查看 Experiment B
8. 如需 OracleTrack，再实现 `supports_oracle_track()`
9. 最后为 Experiment D 协议分别写 TOML，并复用相同的 prepared cache

如果按这个方式做，模型接入层会很薄：

- 数据解析不用你管
- 视频采样不用你管
- frame cache 不用你管
- normalization 不用你管
- 评分不用你管
- summary 不用你管

你只需要负责：

- 如何把 `PreparedSample` 转成模型输入
- 如何把模型输出转成 benchmark 所需格式
