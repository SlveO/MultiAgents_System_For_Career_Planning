# 多模态职业规划助手

本项目研究职业规划场景中的模块化多智能体多模态系统：专用感知Agent负责图像、音频或文档信息提取，文本推理Agent负责职业规划。核心问题是这种协作系统能否达到或超过单体多模态模型，同时保持模块可替换、资源可控和结果可解释。

当前结题MVP流程为：文本或文档输入 → 8个规则追问 → 用户画像 → 职业知识检索 → DeepSeek规划 → 三档反馈 → 脱敏JSONL日志。当前MVP的多模态感知是一次性的；推理Agent向感知Agent发起多轮澄清属于后续研究实验，不应视为已完成功能。

## 设备与依赖

依赖文件是逐层包含关系：

```text
requirements.txt        核心MVP
    ↓
requirements-api.txt    核心MVP + FastAPI/Web API
    ↓
requirements-gpu.txt    API + 本地视觉/语音/向量实验
```

CPU/API机器：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

API/Web机器：

```powershell
pip install -r requirements-api.txt
python -m project.api.run_api
```

带独立显卡或实验室L20的机器：先按CUDA版本安装匹配的PyTorch，再执行：

```powershell
pip install -r requirements-gpu.txt
python scripts/models/download_qwen_vl.py
```

GPU profile同时包含MVP和API能力。模型权重只保存在本地 `models/`，不提交Git。

## CLI演示

配置 `.env` 中的 `DEEPSEEK_API_KEY` 后：

```powershell
python -m project.assistant_cli --session-id demo-1 --goal "六个月内转岗数据分析" --text "我会 Python 和 SQL"
python -m project.assistant_cli --goal "六个月内转岗数据分析" --docs .\examples\sample_profile.txt
```

可用 `--answers-json` 传入固定答案以复现实验。没有API Key或API失败时，程序会使用明确标记的规则模板；模板结果不能计入真实模型实验。

## 测试与实验

```powershell
python -m unittest discover -s project/tests -v
python -m project.experiments.run_completion_experiments --output-dir data/experiments
```

研究实验包括单体多模态模型、单轮模块化系统和多轮感知协作系统的比较，以及知识库、用户追问、感知微调和反馈调整消融。实验设计与指标见 `docs/experiments.md`。

## Docker与Web

Docker使用 `requirements-api.txt`，默认模型为 `deepseek-v4-flash`：

```powershell
docker compose up --build
```

Web前端是可选模块：

```powershell
cd web
npm install
npm run build
```

## Data Layout

`dataset/`保存应版本化的静态职业知识和匿名测试案例；`data/`保存本地数据库、上传文件、日志和实验输出，默认不进入Git。不得提交API Key、个人身份信息、原始上传文件或模型权重。
