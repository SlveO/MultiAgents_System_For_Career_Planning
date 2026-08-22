# 成员 B：L20 执行 Agent 交接

## 任务状态与边界

负责人：成员 B / L20 执行 Agent。

成员 B 原分支 `origin/work/member-b-l20-models@185b55c` 只证明了个人电脑上的
代码与单元测试，不包含 L20 环境、下载、显存或推理证据。不得把原分支中的
占位表格、模型注册表或 Fake 输出写成已完成实验。

本任务必须在联网的 NVIDIA L20 Ubuntu 主机执行。交付物是经过修正的本地模型
适配器、下载/环境脚本、三模型真实冒烟记录和六例预实验结果。权重放在忽略的
`models/`，运行记录放在忽略的 `data/experiments/`。禁止提交权重、缓存、主机名、
账号、令牌、GPU UUID、私人路径或原始环境变量。

## 启动前依赖

负责人先发布包含以下文件的 `origin/work/lead-evaluation` 新提交：

- `project/experiments/architecture_protocol.py`
- `project/experiments/run_architecture_experiments.py`
- `dataset/research_cases/pilot_cases.json` 及六个素材
- 本交接文档

在 L20 仓库根目录执行：

```bash
git fetch origin
git switch -c work/member-b-l20-rework origin/work/lead-evaluation
test -f project/experiments/architecture_protocol.py
test -f dataset/research_cases/pilot_cases.json
git status --short --branch
```

任一文件不存在即停止，并报告“负责人基线尚未发布”。不要从旧的
`integration/week1-results` 直接开始，也不要把成员 B 分支整体合并。只选择性移植：

```bash
git restore --source origin/work/member-b-l20-models -- \
  project/experiments/local_models.py \
  scripts/models/download_research_models.py \
  scripts/models/record_l20_env.py \
  project/tests/test_local_models.py \
  project/tests/test_download_research_models.py \
  project/tests/test_record_l20_env.py
```

## 必须先完成的代码返工

### 1. 主机与环境护栏

`ensure_allowed_platform()` 不能只检查 Linux。增加可测试的 L20 护栏：

- `platform.system()` 必须为 `Linux`；
- `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` 至少有一行名称
  为 `NVIDIA L20`；
- 选定 GPU 的总显存不得低于 45,000 MiB；
- 允许通过参数注入命令输出做单元测试，但正式 CLI 不允许 `--force` 绕过；
- 模型下载前检查仓库所在磁盘至少有 80 GiB 可用空间，不足时停止。

环境记录必须分别保存：Ubuntu、内核、Python、驱动版本、`nvidia-smi` 报告的
CUDA 兼容版本、`nvcc --version` 的本地 Toolkit 状态、PyTorch 版本、
`torch.version.cuda`、BF16 支持、Transformers、ModelScope、磁盘和网络。缺少
`nvcc` 可以记录为“未安装”，不得伪造；不要安装或升级 NVIDIA 驱动。

### 2. 冻结模型与下载清单

在 L20 上重新执行并保存输出：

```bash
git ls-remote https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct.git refs/heads/master
git ls-remote https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct.git refs/heads/master
git ls-remote https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507.git refs/heads/master
```

预期 revision：

| 模型 | 预期 revision |
|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | `5d854aab08710c16b980ec6d603d863b3821b915` |
| `Qwen/Qwen3-VL-2B-Instruct` | `ae9985b208c074c10cfbe3a61b5cb7268cdc9c53` |
| `Qwen/Qwen3-4B-Instruct-2507` | `2de2439ea21be1dc5cb21f22f88af07e43393cbb` |

任一远端值不同即停止，不得自行替换协议。把命令、UTC 时间和结果写入
`data/experiments/environment/model_revisions.json`，交负责人确认后再更新
`dataset/research_protocol.json` 的三个 revision，并在
`docs/research-decisions.md` 追加新决定。

修复下载清单覆盖问题：`--model` 单模型下载时，按 `model_id` 合并已有
`models/research_models_manifest.json`，原有记录不得丢失。每项记录模型 ID、
revision、仓库相对目录、字节数、`config.json` SHA-256 和 UTC 时间。再次读取
清单并确认恰好包含三个唯一模型。

### 3. 推理客户端与三组适配器

把原 `LocalModelAdapter` 收敛为只负责加载/生成的 `LocalModelClient`，另实现：

- `L20MonolithicAdapter(BaseArchitectureAdapter)`；
- `L20ModularOneShotAdapter(BaseArchitectureAdapter)`；
- `L20ModularCollaborativeAdapter(BaseArchitectureAdapter)`；
- `build_l20_adapters(contracts, models_dir, device)` 返回三组映射，供
  `run_architecture_experiments(..., adapters=...)` 注入。

不得在模块顶层导入 `torch`、`transformers`、ModelScope 或 PDF 渲染库。
模型使用 `dtype=torch.bfloat16`、`eval()`、`torch.inference_mode()`、seed 42、
greedy、`do_sample=False`、`num_beams=1`、`max_new_tokens=2048`。不发送
temperature、top-p 或思维链请求。Transformers 必须支持
`Qwen3VLForConditionalGeneration`；旧约束 `transformers<5` 不得直接沿用，先在
L20 验证 `transformers>=4.57,<6`，再把实际成功版本写入环境记录和依赖约束。

所有提示词从 `dataset/research_protocol.json` 原样读取并复核哈希。模型输出只做
`json.loads` 后 Schema 校验：不去 Markdown 围栏、不修复、不重试。无效原文保存
在忽略目录，运行记录标记失败，质量有效分按协议记 1。

图片直接读取版本化素材。PDF 必须只渲染清单指定页，分辨率固定 144 DPI；建议
在 GPU profile 增加 `PyMuPDF` 并把临时页写入 `data/experiments/cache/`。视觉处理
器设置 `min_pixels=256*28*28`、`max_pixels=1280*28*28`，并记录实际视觉 token。
4B 推理器严禁接收图片路径、PDF 路径、像素或原始媒体。

### 4. 冒烟必须验证真实功能

逐个加载、推理、卸载模型，并在两次之间执行显存清理。不得使用“介绍职业规划”
这类与媒体无关的提示词：

1. 8B 使用 `image-001.png`，输出 Schema-valid 规划，并引用至少一条图中事实。
2. 2B 使用同一图片和冻结感知提示词，输出 Schema-valid evidence packet，至少
   正确提取 SQL 或仪表盘事实及位置。
3. 4B 只接收固定画像、知识片段和一份有效 evidence packet，输出
   Schema-valid plan，且 `evidence_used` 引用存在的 evidence ID。

每次记录模型、revision、输入案例、提示词版本/哈希、输出、有效性、延迟、峰值
显存、实际设备和错误。缺一项即不能进入预实验。

## L20 安装与执行顺序

先记录安装前状态，再创建隔离环境：

```bash
python3 scripts/models/record_l20_env.py
python3 -m venv .venv-l20
source .venv-l20/bin/activate
python -m pip install --upgrade pip
```

根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择 Linux、
Pip、Python 和驱动支持的 CUDA wheel，保存实际安装命令。然后：

```bash
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -c "from transformers import Qwen3VLForConditionalGeneration; print('qwen3-vl import ok')"
python -m unittest project.tests.test_architecture_experiments project.tests.test_research_cases -v
python scripts/models/download_research_models.py --model all
python -m project.experiments.local_models --model all --output-dir data/experiments/smoke
```

Qwen3-VL 的加载和 `AutoProcessor` 调用应参照
[Transformers 官方 Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)，
不要根据旧分支猜测 API。

## 两个检查点

### 检查点 A：下载与冒烟

向负责人提交以下摘要后停止，等待确认 revision：

```text
分支与 HEAD：
L20 名称/显存：
Ubuntu/驱动/CUDA Toolkit/PyTorch/Transformers：
三个 revision 是否完全一致：
三个模型目录与总字节数：
三次冒烟 Schema 状态/延迟/峰值显存：
失败或警告：
git status --short：
```

### 检查点 B：六例预实验

负责人确认 revision 已冻结后才运行。为避免重复单体/一次性结果：

```bash
python -m project.experiments.run_architecture_experiments \
  --backend l20 --round-cap 2 --groups all \
  --cases dataset/research_cases/pilot_cases.json \
  --output-dir data/experiments/pilot/cap-2

python -m project.experiments.run_architecture_experiments \
  --backend l20 --round-cap 3 --groups modular_collaborative \
  --cases dataset/research_cases/pilot_cases.json \
  --output-dir data/experiments/pilot/cap-3
```

需要为 CLI 增加上述 `--backend` 和 `--groups`，但 L20 模块必须动态导入。预期
`cap-2` 为六例乘三组共18行，`cap-3` 为协作组6行，共24次真实运行。任何崩溃、
无效 JSON 或显存不足都保留原始记录，不删除失败样本，不进入60次主实验。

## 最终验收

执行并报告：

```bash
python -m compileall -q project
python -m unittest discover -s project/tests -v
git diff --check
git status --short
git ls-files models data
```

验收必须同时满足：默认模块导入不加载 `torch`；三个模型 revision 与协议一致；
环境、清单、冒烟和24次预实验记录齐全；三组输出通过 Schema；协作轮次不超上限；
PDF 为144 DPI；4B未接收媒体；结果包含延迟、峰值显存和失败；`git ls-files
models data` 无权重或运行记录。完成后只提交差异与测试报告给负责人审查，未经明确
授权不得 commit、push、合并或开始20案例主实验。
