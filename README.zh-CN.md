# OmniChainBench 测评框架

这是 OmniChainBench 的评测与数据准备工具。它的核心目标有两个：

1. 把原始 benchmark 数据集转换成可复用的 prepared test data。
2. 以统一、可复现的方式对不同模型进行评测与汇总。

这个框架的设计重点是“先预构建、后复用”。

也就是说，你先对原始视频做一次采样和解码，生成配置里的 `prepared_root` 缓存，例如 `/data/public_data/mllmbenchmark_prepared`；后续评测 10 个 baseline 模型时，直接复用这些缓存，不再重复从视频解码和采样。这样可以显著降低重复开销，也能保证不同模型看到的是完全一致的输入。

## 当前能力

当前版本已经实现：

- 原始数据校验
- Experiment B 所需 chain manifest 生成
- 内置 `main` 协议和可导入自定义协议的 prepared-data 构建
- 在线 adapter 模式评测
- 框架统一负责 prompt 构造、链式历史注入、结构化抽取与评分
- 指标计算与样本级 pass/fail 判断
- 通过 OpenAI-compatible API 的 LLM-as-a-judge
- Experiment A 汇总
- Experiment B 汇总，包括 OracleTrack rerun

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
- [src/omnichain_eval/prompting.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/prompting.py)：模型输入 prompt 渲染和链式历史构造
- [src/omnichain_eval/normalize.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/normalize.py)：结构化输出严格校验
- [src/omnichain_eval/structurer.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/structurer.py)：固定 structurer LLM 接口
- [src/omnichain_eval/template_pack.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/template_pack.py)：通用 Markdown prompt 模板加载
- [src/omnichain_eval/metrics.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/metrics.py)：指标计算
- [src/omnichain_eval/judge.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/judge.py)：Judge 接口
- [src/omnichain_eval/experiments.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/experiments.py)：实验编排
- [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py)：模型 adapter 接口
- `prompts/benchmark_v1/`：任务级 inference prompt 模板
- `prompts/benchmark_oracle_v1/`：OracleTrack 上游 inference prompt 基目录，内部包含 `language/`、`visual/`、`language_visual/` 三套模板
- `prompts/structurer_v1/`：任务级 structurer prompt 模板
- `prompts/structurer_oracle_v1/`：OracleTrack 上游 structurer prompt 模板
- `prompts/judge_v1/`：任务级 judge prompt 模板目录

当前 prompt 模板约定：

- 每个 Markdown 文件本身就是最终 prompt 正文
- 不再使用 `# system` / `# user` 这种分段格式
- benchmark、structurer、judge 在运行时都只发送 user prompt
- `prompts/benchmark_v1/` 和 `prompts/structurer_v1/` 都覆盖 10 个 benchmark 任务
- `prompts/judge_v1/` 覆盖 9 个需要 judge 的任务；`Spatial_Temporal_Grounding` 是规则评分，不走 judge prompt
- `prompts/benchmark_oracle_v1/` 只覆盖 OracleTrack 上游 rerun 的 `Continuous_Actions_Caption` 与 `Spatial_Temporal_Grounding`，并按 `language/`、`visual/`、`language_visual/` 三个子目录组织
- `prompts/structurer_oracle_v1/` 只覆盖 OracleTrack 上游 structurer
- `configs/examples/`：常见流程的 TOML 示例配置

运行后产生的目录：

- 配置中的 `prepared_root`，例如 `/data/public_data/mllmbenchmark_prepared/`：预构建后的测试数据缓存
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

说明：

- 在当前环境下建议显式设置 `UV_CACHE_DIR=/tmp/uv-cache`，避免默认缓存目录不可写。
- 视频解码只使用 PyAV；如果环境里没有 PyAV，`prepare-data` 会直接报错。

## 数据集假设

框架要求你在 TOML 中显式配置原始数据根目录。在当前服务器示例里，这个目录是 `/data/public_data/mllmbenchmark`。

目录结构至少满足：

- 主标注：`<data_root>/<sport>/<event>/<video_id>.json`
- 视频：`<data_root>/<sport>/<event>/<video_id>.mp4`
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
- protocol spec
- adapter 路径
- inference prompt 路径
- prepare-data 的 worker 数
- structurer backend 与 prompt 路径
- judge backend 与 prompt 路径

这样你就可以为不同实验、不同协议、不同模型分别维护不同的 TOML 文件。

仓库内提供了这些示例配置：

- [configs/examples/workflow.toml](/home/qi7876/dev/eval-tools/configs/examples/workflow.toml)：数据校验与 chain manifest 生成
- [configs/examples/prepare_main.toml](/home/qi7876/dev/eval-tools/configs/examples/prepare_main.toml)：只构建 `main` 协议缓存
- [configs/examples/prepare_custom_protocol.toml](/home/qi7876/dev/eval-tools/configs/examples/prepare_custom_protocol.toml)：自定义协议的 `prepare-data` 示例
- [configs/examples/run_eval_adapter.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_adapter.toml)：mock smoke test 评测
- [configs/examples/run_eval_main.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_main.toml)：`main` 协议在线评测示例
- [configs/examples/run_eval_custom_protocol.toml](/home/qi7876/dev/eval-tools/configs/examples/run_eval_custom_protocol.toml)：自定义协议的 `run-eval` 示例

支持的顶层 section：

- `[validate_data]`
- `[build_chain_manifest]`
- `[prepare_data]`
- `[run_eval]`
- `[structurer]`
- `[judge]`

最小 `run-eval` 示例：

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

路径规则：

- 相对路径都以当前 TOML 文件所在目录为基准解析
- 绝对路径可以直接写
- `protocol` / `protocols` 既可以写内置 id，例如 `main`，也可以写可导入 Python 类，例如 `your_package.protocols:EightFrameUniformProtocol`
- 不同实验建议拆成不同 TOML 文件
- `run-eval` 相关的 `[run_eval]`、`[structurer]`、`[judge]` 建议放在同一个 TOML 中统一管理
- 密钥更建议通过 `api_key_env` 指向环境变量

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
- 解析 MOT tracking 文件
- 校验任务 schema
- 校验 `Spatial_Imagination` 是否带有合法的 `upstream_annotation_id`

返回行为：

- 支持任务没有校验问题时退出码为 `0`
- 只要支持任务存在校验问题就退出码为 `1`
- 终端打印前 50 个问题

重要说明：

- 原始数据里不受支持的任务会单独统计并忽略，不会阻断主评测流程
- 当前实现按支持任务的 annotation 粒度容错
- 某个支持任务 sample 损坏时，只会跳过该 sample，不会整份 JSON 全部失效

## 第二步：生成 Experiment B 的 chain manifest

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  build-chain-manifest \
  --config configs/examples/workflow.toml
```

这个命令会：

- 找到所有受支持的 `Spatial_Imagination`
- 读取其 `upstream_annotation_id`
- 在同一个源 annotation 文件内定位上游 sample
- 校验上游任务是否是：
  - `Continuous_Actions_Caption`
  - `Spatial_Temporal_Grounding`
- 生成 Experiment B 所需的显式 pair 列表

重要说明：

- 当前不支持的任务会在扫描阶段被忽略，不会阻断 chain manifest 生成

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

### 构建主协议缓存

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval \
  prepare-data \
  --config configs/examples/prepare_main.toml
```

仓库内提供的示例配置会直接构建可复用的 `main` prepared cache。
如果某个模型需要不同于 `main` 的原生采样方式，那么应该把 `[prepare_data].protocols` 指向一个可导入的自定义协议类。

这个命令会：

- 按当前配置的协议对 sample 进行采样
- 解码相应视频帧
- 把每个 sample 写成一个独立 bundle
- 当 `[prepare_data].media_formats` 包含 `sampled_video` 时，再额外基于采样帧编码一个 sampled MP4
- 当 `[prepare_data].generate_oracle_visual_media = true` 时，再额外生成 Oracle 视觉注入所需的画框版媒体
- 存储 sampled-to-original 映射
- 存储任务评测需要的 GT sidecar
- 当 `[prepare_data].workers > 1` 时，会按视频并发处理

重要说明：

- 只有当前支持的 benchmark task 会进入 prepared-data
- 不支持的任务会被忽略，并记录到协议元数据里
- 支持任务本身如果有校验错误，`prepare-data` 仍然会直接失败
- `workers` 控制单个 protocol 内部按视频并发的线程数；不同 protocol 之间仍然串行构建
- 仓库自带的 prepare 示例都使用 `media_formats = ["frames", "sampled_video"]`
- `configs/examples/prepare_main.toml` 还会开启 `generate_oracle_visual_media = true`，用于 Experiment B 的 Oracle 视觉/联合实验
- `configs/examples/prepare_custom_protocol.toml` 展示了如何引用一个可导入的自定义协议类
- sampled video 只会从已经采样好的帧重编码，不会直接截原视频；只给多帧 sample 生成，编码为无音频 H.264 (`libx264`)

### 为什么要先构建 prepared data

因为 benchmark 中多个模型会共享完全相同的采样输入。若每次评测模型时都重新从原视频读取和采样，会有几个问题：

- 重复解码视频，开销大
- 不同模型运行时容易出现输入不一致
- 不便于复现实验
- OracleTrack / chain rerun 等流程会更难统一

prepared-data 方案解决了这些问题。

## 支持的协议形式

当前框架支持两类 protocol spec：

- 内置 id，例如 `main`
- 可导入 Python 类，例如 `your_package.protocols:EightFrameUniformProtocol`

说明：

- `main` 仍然是框架内置 benchmark 协议，覆盖全部任务，包括 STG
- 自定义协议就是表达模型原生采样方式的入口，例如固定 8 帧均匀采样、按 1 fps 采样等
- protocol 只负责采样；prompt、structurer、judge、metrics、chain 和 OracleTrack 仍然由框架统一负责
- `prepare-data` 和 `run-eval` 必须使用同一个 protocol spec；不一致时会直接报错并要求重建 prepared data

### 自定义协议接口

自定义协议需要继承 [src/omnichain_eval/protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/protocols.py) 中的 `BaseProtocol`。
仓库内也提供了一个可直接导入的示例模块 [src/omnichain_eval/example_protocols.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/example_protocols.py)，供示例 TOML 使用。

最小示例：

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

然后在 TOML 中这样引用：

```toml
[prepare_data]
protocols = ["omnichain_eval.example_protocols:ExampleEightFrameUniformProtocol"]

[run_eval]
protocol = "omnichain_eval.example_protocols:ExampleEightFrameUniformProtocol"
```

## Prepared Data 的目录结构

执行 `prepare-data` 后，每个解析出来的协议都会在 `prepared_root` 下拥有独立目录：

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
- `sampled_video_file`
- `sampled_video_fps`
- `oracle_visual_frame_files`
- `oracle_visual_sampled_video_file`
- `reference_payload`
- `q_window` 或 `a_window`
- `upstream_annotation_id`
- `metadata`

`manifest.json` 里保存的 `frame_files` 和 `sampled_video_file` 都是相对 bundle 的路径，运行时 `load_prepared_samples()` 会统一改写成绝对路径。
`sampled_video_file` 只会在多帧 sample 且 prepare 时启用了 sampled video 时出现。
`sampled_video_fps` 表示采样后这段输入对应的大致播放帧率，adapter 和 benchmark prompt 都可以直接使用它。
如果 prepare 时启用了 Oracle 视觉媒体生成，那么 Oracle 上游样本还会额外带上 `oracle_visual_frame_files` 和 `oracle_visual_sampled_video_file`。
运行时，`metadata` 里还会补充 `protocol_spec` 和 `protocol_manifest`。

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

- `protocol_id`
- `protocol_spec`
- `protocol_manifest`
- `data_status`，其中包含 raw dataset 计数、supported dataset 计数、ignored unsupported task 计数、supported issue 计数
- `supported_dataset_fingerprint`
- prepared sample 数量

此外，`stats.json` 也会额外写入 `ignored_unsupported_sample_count` 和 `ignored_unsupported_task_counts`。

## 第四步：运行评测

当前 `run-eval` 只保留在线 adapter 模式。

每个 run 目录下会维护 6 个可续测文件：

- `predictions.jsonl`：独立任务和链式上游任务的原始输出
- `structured_predictions.jsonl`：这些样本的结构化输出
- `results.jsonl`：这些样本的完成态评测结果
- `chain_predictions.jsonl`：链式下游任务的原始输出
- `chain_structured_predictions.jsonl`：链式下游任务的结构化输出
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

[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
```

或者：

```toml
[run_eval]
adapter = "my_project.adapters.qwen:QwenVideoAdapter"
prompt_root = "prompts/benchmark_v1"

[structurer]
backend = "openai"
prompt_root = "prompts/structurer_v1"
```

`[run_eval].prompt_root` 现在是必填项，必须指向一个包含 10 个任务 Markdown 模板的 prompt 目录。
`[structurer].prompt_root` 也是必填项，必须指向 structurer prompt 模板目录。
如果 `[run_eval].enable_oracle_track = true`，那么 `[run_eval].oracle_prompt_root` 和 `[structurer].oracle_prompt_root` 也都是必填项。
其中 `[run_eval].oracle_prompt_root` 必须指向包含 `language/`、`visual/`、`language_visual/` 三个子目录的 Oracle prompt 基目录。
`[structurer].oracle_prompt_root` 则指向 Oracle 上游 structurer prompt 目录。

### adapter 接口

你需要继承 [src/omnichain_eval/adapters/base.py](/home/qi7876/dev/eval-tools/src/omnichain_eval/adapters/base.py) 中的 `BaseModelAdapter`。

接口如下：

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

### adapter 会收到什么

adapter 现在会收到一个 `ModelInput`。常用字段包括：

- `model_input.messages`
- `model_input.sample`

其中 `model_input.sample` 里常见字段有：

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
- `q_window`
- `a_window`
- `metadata`

通常你把 `model_input.messages` 直接送给模型，然后根据模型能力从 `model_input.sample` 里选媒体输入：

- 图像模型使用 `frame_files`
- 原生视频模型优先使用 `sampled_video_file`
- 如果模型接口需要显式的时间信息，可以直接读取 `sampled_video_fps`

说明：

- `model_input.sample.frame_files` 是已经展开后的绝对路径
- `model_input.sample.sampled_video_file` 如果存在，也已经是绝对路径
- 每张图已经按采样顺序排好
- sampled 索引就是 `frame_files` 的顺序索引
- 当前 protocol 已经决定了模型真正看到哪些帧或 sampled video；adapter 不应自行重新采样
- 如果模型需要不同于当前 protocol 的原生采样方式，应新增 `BaseProtocol` 子类并重新生成 prepared data

如果当前 sample 是链式下游 `Spatial_Imagination`，框架会自动把最终消息构造成：

- `user`：上游问题
- `assistant`：上游答案
- `user`：当前任务模板渲染出的下游 prompt

### adapter 应返回什么

adapter 应直接返回模型的原始回答字符串。

后续工作都由框架统一处理：

- 把原始回答送入固定的 structurer 模块
- 校验结构化 JSON
- 再把校验后的结构化结果送入评分和 judge 逻辑

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
  "objects": [
    {"label": "Player A", "bbox": [xtl, ytl, xbr, ybr]},
    {"label": "Player B", "bbox": [xtl, ytl, xbr, ybr]}
  ]
}
```

在 structurer / validation 阶段，OSR 仍然按 label 强约束。
如果某个必需 label 整条缺失，框架会按 GT label 顺序补入该对象，并使用 sentinel bbox `[-1, -1, -1, -1]`，以便继续做按 label 对齐的 IoU 计算。
如果该 label 已经出现，框架不会再二次修补它的 bbox；不符合 schema 的行仍会直接导致 structuring 失败。
文本关系仍然正常进入 judge，而 sentinel 对象的 IoU 会记为 `0`。

#### `Continuous_Events_Caption`

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

        # 在这里替换成你的真实模型调用逻辑
        if sample.task_name == "Scoreboard_Single":
            return '{"text": "The score is 1-0.", "bbox": [100, 900, 1000, 980]}'

        if video_path is not None:
            _ = video_path
        else:
            _ = image_paths
        _ = messages
        return '{"text": "placeholder"}'
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
prepared_root = "/data/public_data/mllmbenchmark_prepared"
protocol = "main"
artifacts_root = "artifacts/runs"
prompt_root = "prompts/benchmark_v1"
adapter = "my_package.my_adapter:MyVideoAdapter"

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
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-397b-a17b"
temperature = 0
invalid_json_retries = 2
concurrency = 1

[structurer.extra_body]
enable_thinking = false
```

## Structurer 配置

Structurer 的配置写在 `run-eval` 所使用 TOML 的 `[structurer]` section 里。
`[structurer].prompt_root` 必须指向一个目录，目录里要为每个任务提供一个 Markdown 模板。
每个 Markdown 文件本身就是发送给 structurer 模型的最终 user prompt。

典型 OpenAI-compatible 配置如下：

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

`[structurer].extra_body` 会被原样透传到 OpenAI-compatible 请求体里。
`[structurer].temperature` 是可配置的；默认值是 `0`。
如果你不手动覆盖，框架默认会使用 `model = "qwen3.5-397b-a17b"`、`temperature = 0` 和 `extra_body.enable_thinking = false`。

如果只是本地 smoke test，也可以不调用外部 structurer API，直接使用：

```toml
[structurer]
backend = "static-parse"
prompt_root = "prompts/structurer_v1"
invalid_json_retries = 1
concurrency = 1
```

当前 structurer 的行为约束：

- prompt 是按任务细写的，不再是一套完全通用的模板
- structurer 可以对 raw output 里的显式值做轻度整理
- 它根据当前任务的 schema 和 prompt 模板来判断应整理出哪些标准字段
- 如果 raw output 里同时有分析过程和最终答案，structurer 应优先整理最终答案
- structurer 不应凭空补出 raw output 里没有出现的 bbox、区间、tracking 或答案文本
- 对 `Scoreboard_Single`，框架会接受 structurer 显式输出的 sentinel bbox `[-1, -1, -1, -1]`，但不会自动补缺失的 bbox 字段
- 对 `Objects_Spatial_Relationships`，框架会按 GT label 重新排序，并对整条缺失的必需 label 自动补一个 sentinel bbox `[-1, -1, -1, -1]`，但不会修补已经出现 label 的 bbox 字段

当前 retry 规则：

- structurer 返回内容如果不是合法 JSON，会自动重试
- structurer 返回内容即使是合法 JSON，只要不满足该任务的严格结构校验，也会自动重试
- 重试次数由 `[structurer].invalid_json_retries` 控制
- 如果所有尝试都失败，该样本会留到下一轮继续，本轮不会写入结构化产物
- Structurer 在后台异步执行；`[structurer].concurrency` 控制并发 worker 数

## Judge 配置

Judge 的配置写在 `run-eval` 所使用 TOML 的 `[judge]` section 里。
`[judge].prompt_root` 必须指向一个目录，目录里要为每个进入 judge 的任务提供一个 Markdown 模板。
每个 Markdown 文件本身就是发送给 judge 模型的最终 user-only prompt。

典型配置如下：

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

`[judge].extra_body` 也会原样进入请求体。
`[judge].temperature` 是可配置的；默认值是 `0`。
如果你不手动覆盖，框架默认会使用 `model = "qwen3.5-397b-a17b"`、`temperature = 0` 和 `extra_body.enable_thinking = false`。

然后只在环境变量里提供密钥：

```bash
export DASHSCOPE_API_KEY="your-api-key"
```

当然，你也可以把 `api_key` 直接写进 TOML，但更推荐用 `api_key_env`。如果 judge 和 structurer 都走 DashScope，它们可以共用同一个 `DASHSCOPE_API_KEY`。

如果只是本地 smoke test，不想真的调 judge API，可以把 backend 改成：

```toml
[judge]
backend = "static-pass"
prompt_root = "prompts/judge_v1"
```

或者：

```toml
[judge]
backend = "static-fail"
prompt_root = "prompts/judge_v1"
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
- 这些样本的结构化结果写入 `structured_predictions.jsonl`
- 这些样本的完成态评测写入 `results.jsonl`
- 链式下游任务的推理结果写入 `chain_predictions.jsonl`
- 链式下游任务的结构化结果写入 `chain_structured_predictions.jsonl`
- 链式下游任务的完成态评测写入 `chain_results.jsonl`
- 如果中途中断，再次使用同一个配置运行时，会自动跳过已完成样本
- 如果恰好中断在“预测已写入、结构化未完成”，则会复用已保存的 prediction，只补做结构化
- 如果恰好中断在“结构化已写入、评分未完成”，则会复用已保存的结构化结果，只补做评分
- 如果链式上游样本还没有产出回答，对应下游样本会保持 blocked，等下一轮继续
- 续测状态不是靠单独的失败清单，而是每轮都重新比对这 6 个 artifact 文件

重要说明：

- 断点续测依赖于写入同一个 run 目录
- 实际使用时，建议为长任务显式设置固定的 `[run_eval].run_name`
- OracleTrack 的 pair 级评测会分别通过
  `oracle_language_pair_results.jsonl`、
  `oracle_visual_pair_results.jsonl`、
  `oracle_language_visual_pair_results.jsonl`
  独立断点恢复

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
- `structured_predictions.jsonl`
- `results.jsonl`
- `chain_predictions.jsonl`
- `chain_structured_predictions.jsonl`
- `chain_results.jsonl`
- `summary.json`

这些文件的语义是：

- `predictions.jsonl`：独立任务和链式上游任务的原始模型输出
- `structured_predictions.jsonl`：这些样本校验通过的结构化输出
- `results.jsonl`：这些样本的完成态评测结果
- `chain_predictions.jsonl`：链式下游任务的原始模型输出
- `chain_structured_predictions.jsonl`：链式下游样本校验通过的结构化输出
- `chain_results.jsonl`：链式下游任务的完成态评测结果
- 如果某个 sample 已经完成预测，但结构化还没完成，那么它不会写入对应的 `*_structured_predictions.jsonl`，下一轮会继续补测
- 如果某个 sample 已经完成结构化，但 judge 或评分流程没有完成，那么它不会写入对应的 `*_results.jsonl`，下一轮会继续补测

`summary.json` 会额外记录：

- `data_status`，直接继承自当前协议的 `build_manifest.json`
- 本轮目标 sample 总数
- 本轮开始前已完成数、本轮新完成数、累计完成数
- `pending_prediction_sample_ids`
- `predicted_not_structured_sample_ids`
- `structured_not_evaluated_sample_ids`
- `overall`
- 各 task 的 summary
- `experiment_b`
- `pending_chain_prediction_sample_ids`
- `chain_predicted_not_structured_sample_ids`
- `chain_structured_not_evaluated_sample_ids`
- `blocked_chain_sample_ids`
- 本轮 normal / chain / oracle 失败摘要

unsupported task 的信息只放在 `data_status` 中，不会混入 runtime 的 pending 计数，也不会进入 task accuracy。

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

当前 `experiment_b` 会按分组输出：

- `base`：understanding accuracy、reasoning accuracy、chain success、chain success (w/o track)
- `oracle_language`：只做语言 GT tracking 注入的 text-only Oracle rerun
- `oracle_visual`：只做视觉 GT tracking 注入的 text-only Oracle rerun
- `oracle_language_visual`：同时做语言和视觉注入的 text-only Oracle rerun

每个 Oracle 分组都会输出：

- `num_chain_samples`
- `num_scored_chain_samples`
- `num_pending_chain_samples`
- `understanding_acc`
- `reasoning_acc`
- `chain_success_wo_track`

## OracleTrack

OracleTrack 现在完全由框架实现。

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
- 使用 `[run_eval].oracle_prompt_root` 作为三套 Oracle prompt 基目录，`[structurer].oracle_prompt_root` 作为 Oracle 上游 structurer prompt
- 依次运行 `language`、`visual`、`language_visual` 三组 Oracle 变体
- `language` 变体会在 upstream Oracle prompt 正文中直接注入 GT tracking
- `visual` 变体会把 upstream 输入媒体切换成已经画好 GT tracking 框的图片/视频
- `language_visual` 变体会同时使用两种注入方式
- 明确告诉模型主体已经由 GT tracking 指定，因此 Oracle upstream 输出不需要再生成 tracking
- 下游输入会重新基于“上游完整渲染 prompt + 上游原始回答”构建链式历史
- Oracle upstream 评分只看非 tracking 分量，Experiment B 中额外汇总 Oracle 的 text-only chain 指标

## BERTScore

只要任务提供了可比较的文本输入，评测时就会始终计算 BERTScore。

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
- `Scoreboard_Single` 的显式 sentinel bbox，以及 `Objects_Spatial_Relationships` 中显式或由缺失 label 自动补出的 sentinel bbox，都会让对应 IoU 记为 `0`
- 原始数据中的 unsupported task 会被忽略，并记录到 `build_manifest.json` / `summary.json`
- 支持任务本身的数据校验错误仍然会阻断 `prepare-data`

## 运行测试

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

当前测试覆盖：

- 协议采样
- chain manifest 生成
- prepared-data 构建
- mock adapter 评测
- structurer 重试和结构校验逻辑
- prediction -> structuring -> evaluation 的断点续测
- Experiment B 汇总流程

## 常用命令模板

### 全流程 smoke test

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval build-chain-manifest --config configs/examples/workflow.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --config configs/examples/prepare_main.toml
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval run-eval --config configs/examples/run_eval_adapter.toml
```

## 当前限制

- 还没有内置 10 个 baseline 的具体 adapter
- 还没有内置一套 baseline 专属的原生采样协议类，需要你按模型自行实现
- prepared-data 当前按 sample 写 JPEG bundle，没有做共享帧去重存储

## 推荐接入新模型的方式

建议按以下顺序接入一个新模型：

1. 先为这个模型和协议新建一个独立 TOML 文件
2. 如果模型需要不同于 `main` 的原生采样方式，先实现一个 `BaseProtocol` 子类
3. 运行 `prepare-data`
4. 编写一个 `BaseModelAdapter` 子类
5. 在 TOML 里同时配置 `[run_eval].prompt_root` 和 `[structurer].prompt_root`
6. 让 adapter 只消费 `model_input.sample`、`model_input.messages` 和 sample 对应的 prepared frame bundle
7. 让 adapter 直接返回模型原始回答字符串
8. 运行 `run-eval --config your_model.toml`，并确保它使用的是与你 prepare 时相同的 protocol
9. 在 TOML 中设置 `[run_eval].chain_manifest` 查看 Experiment B
10. 如需 OracleTrack，在 TOML 中设置 `[run_eval].enable_oracle_track = true`

如果按这个方式做，模型接入层会很薄：

- 数据解析不用你管
- 视频采样不用你管
- frame cache 不用你管
- structurer 和结构校验不用你管
- 评分不用你管
- summary 不用你管

你只需要负责：

- 如何把 `model_input.sample.frame_files` 或 `model_input.sample.sampled_video_file` 和 `model_input.messages` 送进真实模型
- 如何把模型原始回答字符串返回给框架
