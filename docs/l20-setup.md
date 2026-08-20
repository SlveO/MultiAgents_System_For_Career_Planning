# L20 主机环境记录与本地研究模型(成员B)

状态:**代码与文档已交付,环境记录/下载/冒烟推理等待 L20 访问后执行。**
本文档不含主机名、账号、令牌、私人路径或任何身份信息。

## 边界规则(必须遵守)

- 研究模型权重只在可联网的 NVIDIA L20 Ubuntu 主机(48 GB 显存档)下载与运行;
- **严禁**在当前 Windows 开发机上执行下载脚本或主实验推理;
- `scripts/models/download_research_models.py` 内置平台护栏,非 Linux 直接拒绝;
- 权重放在已忽略的 `models/`,推理结果放在已忽略的 `data/experiments/`;
- 模板/离线假模型结果不得计入真实模型实验(docs/experiments.md)。

## 冻结模型(2026-08-20 固定,ModelScope master commit)

| 模型 | 组 | Revision (pin) |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | A 单体多模态 | `5d854aab08710c16b980ec6d603d863b3821b915` |
| `Qwen/Qwen3-VL-2B-Instruct` | B/C 感知 | `ae9985b208c074c10cfbe3a61b5cb7268cdc9c53` |
| `Qwen/Qwen3-4B-Instruct-2507` | B/C 推理 | `2de2439ea21be1dc5cb21f22f88af07e43393cbb` |

Revision 来源:`git ls-remote https://www.modelscope.cn/models/<id>.git refs/heads/master`
(2026-08-20 查询)。下载脚本与 `project/experiments/local_models.py` 共用同一注册表。

## L20 环境记录(第 1 天:安装任何东西之前执行)

在 L20 Ubuntu 主机仓库根目录运行:

```bash
python scripts/models/record_l20_env.py
```

脚本自动脱敏(主机名/用户名/私人路径/密钥形态)后输出一张 markdown 表格,
同时把原始(已脱敏)记录写入忽略目录 `data/l20/`。将表格粘贴到下面替换占位符:

| 项目 | 记录 |
|---|---|
| Ubuntu 版本 | *(待 L20 执行)* |
| 内核 (uname -a, 主机名已脱敏) | *(待 L20 执行)* |
| Python | *(待 L20 执行)* |
| 驱动 / 显存 (nvidia-smi) | *(待 L20 执行)* |
| PyTorch / CUDA | *(待 L20 执行)* |
| 磁盘 (df -h, 私人路径已脱敏) | *(待 L20 执行)* |
| 网络 (modelscope.cn) | *(待 L20 执行)* |

## 安装步骤(L20 Ubuntu)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# 先装与本机 CUDA 匹配的 PyTorch(见 pytorch.org 的 CUDA 版本指引),例如:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

只有上述预检通过后,才允许执行下载。

## 下载(第 2 天)

```bash
python scripts/models/download_research_models.py --model all
```

脚本按固定 revision 下载到忽略的 `models/`,并写
`models/research_models_manifest.json`(模型、revision、体积、UTC 时间)。
单模型:`--model Qwen/Qwen3-VL-2B-Instruct`。

## 确定性冒烟推理(第 2 天,每个模型一次)

```bash
python -m project.experiments.local_models --model all
```

greedy + seed 42,不采样;输出文本、耗时、设备与**峰值显存 (peak_vram_mb)**,
JSON 结果写入 `data/experiments/smoke/`。8B 单模型:
`--model Qwen/Qwen3-VL-8B-Instruct`。

## 验收清单(对照 docs/progress.md 成员 B 行)

- [ ] 三个模型 revision 与本文冻结表一致(manifest 与冒烟输出可核);
- [ ] 每次冒烟记录峰值显存;
- [ ] 文档与记录无主机名、账号、令牌、私人路径;
- [ ] Windows 机器上从未发生模型下载;
- [ ] 分支 `work/member-b-l20-models` 已推送供负责人审查。

## 阻塞记录(负责人规则:记录准确命令/错误/证据/下一步,不得用未验证结果替代)

- **阻塞项**:L20 主机访问未就绪(截至 2026-08-20,成员 B 无可用的 L20 SSH 条件)。
- **已执行命令**:`git ls-remote https://www.modelscope.cn/models/<id>.git refs/heads/master`(仅元数据,非权重),已取得并固定三模型 revision。
- **错误**:无(未尝试在 Windows 下载权重)。
- **已完成证据**:下载脚本、适配器、环境记录脚本及单元测试均已在 Windows/无 GPU 环境验证通过(见提交)。
- **下一步动作**:取得 L20 访问后,依次执行本文件"环境记录 → 安装 → 下载 → 冒烟"四节,回填环境表格并把 manifest/冒烟 JSON 附给负责人审查。
