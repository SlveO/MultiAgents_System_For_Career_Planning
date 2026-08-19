# 多模态职业规划助手

本项目研究一个明确问题：在职业规划场景中，由多个专注单一模态的感知 Agent 与文本推理 Agent 组成的模块化系统，能否达到或超过单体多模态模型。职业规划是应用和评测场景，系统架构对比才是主研究变量。

当前已完成的结题 MVP 是：

```text
文本或文档输入
-> 8 个固定追问
-> 结构化用户画像
-> 65 条职业知识检索
-> DeepSeek 规划或明确标记的模板降级
-> 30/90/180 天路线
-> 过短/合适/过于详细反馈
-> 脱敏 SQLite 与 JSONL 记录
```

当前感知流程是一次性的。推理 Agent 主动向感知 Agent 多轮索取证据，以及与单体多模态模型的正式对照实验，属于后续研究阶段，不能视为已实现。

## 安装

依赖按设备逐层包含：

```text
requirements.txt        核心文本/文档 MVP
    -> requirements-api.txt    MVP + FastAPI/Web
        -> requirements-gpu.txt    API + 本地视觉/语音/向量实验
```

默认环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

API/Web 机器使用 `pip install -r requirements-api.txt`。独显或实验室 L20 机器应先安装与本机 CUDA 匹配的 PyTorch，再安装 `requirements-gpu.txt`。GPU 配置同时拥有 API 和 MVP 功能；模型权重只放在已忽略的 `models/`。主研究模型只在可联网的 L20 Ubuntu 主机下载和运行，当前开发机器不下载这些权重。

## CLI 演示

交互执行 8 个追问：

```powershell
python -m project.assistant_cli --session-id demo-1 --goal "六个月内转岗数据分析" --text "我会 Python 和 SQL"
python -m project.assistant_cli --goal "获得后端实习" --docs .\examples\sample_profile.txt
```

支持 `txt`、`md`、`csv`、`tsv`、`pdf`、`docx`、`xlsx`，不声明支持旧式 `.doc` 或 `.xls`。可用 `--answers-json` 固定追问答案，或用 `--no-follow-up` 运行消融。`project/main.py` 和仓库根 `main.py` 仅作兼容入口，新代码统一使用 `project.assistant_cli`。

将 `DEEPSEEK_API_KEY` 写入本地 `.env` 后使用 `deepseek-v4-flash`；请求明确发送 `thinking={"type":"disabled"}`。没有密钥、响应无效或服务失败时会降级为 `local_fallback`。模板和离线假模型结果不得计入真实模型实验。

## 测试与实验

```powershell
python -m compileall -q project
python -m unittest discover -s project/tests -v
python -m project.assistant_cli --help
python -m project.experiments.run_completion_experiments --output-dir data/experiments
```

默认实验使用确定性的 Fake DeepSeek，只验证四组结题对照的流程和结果文件，共输出 8 行；真实 API、GPU、前端与 Docker 的手工步骤和验收标准见 [验证指南](docs/verification.md)。主研究实验设计见 [实验设计](docs/experiments.md)。

启动可选 API：

```powershell
pip install -r requirements-api.txt
python -m project.api.run_api
```

前端位于 `web/`，修改后运行 `npm install` 和 `npm run build`。Docker 使用 `requirements-api.txt`：`docker compose up --build`。

## 目录与协作

- `project/`：唯一正式 Python 实现和测试。
- `dataset/`：应版本化的知识库与匿名案例。
- `data/`：本地日志、数据库、上传和实验输出，不提交。
- `models/`：本地模型权重，不提交。
- `corwork/`：本地成员成果审查区，不作为第二套源码。
- `docs/`：架构、接口、实验、验证及中英文进度。

不得提交 API Key、原始简历、姓名、学号、电话、邮箱、账号、私人路径、缓存或生成日志。当前阶段和四人交接任务见 [团队进度](docs/progress.zh-CN.md)；Agent 执行状态见 [Project Progress](docs/progress.md)。

当前团队统一基线为 `origin/integration/week1-results`。每位成员先同步该分支，再创建自己的任务分支；不要直接向共享整合分支并发提交：

```powershell
git fetch origin
git switch -c work/<role>-<task> origin/integration/week1-results
```

2026-08-20 至 2026-08-21 的负责人、交付物、依赖和验收标准见[团队进度](docs/progress.zh-CN.md)。`main` 暂不自动合并，待两天成果审查后再单独决定。
