# 多模态职业规划助手（结题 MVP）

本仓库面向大学生创新项目结题演示，主流程为：文本或文档输入 → 8 个固定追问 → 结构化用户画像 → 职业知识检索 → DeepSeek 生成 30/90/180 天规划 → 三档反馈 → 脱敏 JSONL 日志。

默认配置固定为：

- API：`https://api.deepseek.com`
- 模型：`deepseek-v4-flash`
- 模式：`thinking={"type":"disabled"}`
- 密钥：环境变量 `DEEPSEEK_API_KEY`

图片、音频、视频、向量检索、FastAPI、Web 和 GPU 模型均为可选功能，不影响文本与文档主流程。

## 快速开始

建议使用 Python 3.10 或 3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-mvp.txt
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`，然后运行唯一 CLI 入口：

```powershell
python -m project.assistant_cli --session-id demo-1 --goal "获得数据分析实习" --text "会 Python 和 SQL"
```

CLI 会依次询问 8 个问题，并在规划结束后要求选择“过短 / 合适 / 过于详细”。无 API Key 或请求失败时会回退到本地规则模板，便于离线演示。

## 输入与输出

文档明确支持：`.txt`、`.md`、`.csv`、`.tsv`、`.pdf`、`.docx`、`.xlsx`。例如：

```powershell
python -m project.assistant_cli --goal "六个月内转岗数据分析" --docs .\examples\sample_profile.txt
```

可重复演示时，可跳过交互追问并传入 JSON：

```powershell
python -m project.assistant_cli --goal "获得数据分析实习" --answers-json '{"education":"本科大三","major":"统计学","skills":"Python、SQL","interests":"数据分析","target_role":"数据分析师","time_budget":"每周10小时","preference":"上海互联网","constraints":"项目经验不足"}'
```

反馈完成后，脱敏运行记录写入 `data/logs/runs.jsonl`。日志不保留原始文档路径、完整文档内容、姓名、手机号、邮箱、学号、身份证号或 Windows 用户名。

## 实验

以下命令离线运行四组对照实验，不调用真实 API：

```powershell
python -m project.experiments.run_completion_experiments --output-dir data/experiments
```

实验覆盖规则模板 vs DeepSeek、无检索 vs 有检索、仅文本 vs 文本加文档、无追问 vs 有追问，并生成 `results.json` 与 `results.csv`。指标包括完整性、个性化、可执行性、响应时间和反馈。

需要生成真实 DeepSeek 对照结果时，在配置 API Key 后显式增加 `--live`；该模式会产生 API 调用与费用：

```powershell
python -m project.experiments.run_completion_experiments --output-dir data/experiments-live --live
```

## 测试

```powershell
python -m unittest discover -s project/tests -v
python -m unittest test_model.test_mvp_components -v
```

前端为可选模块；需要验证时运行：

```powershell
cd web
npm install
npm run build
```

## 可选服务

安装完整依赖后可启动 FastAPI：

```powershell
python -m project.api.run_api
```

打开 `http://localhost:8000`。部署时应显式设置强随机 `JWT_SECRET_KEY` 并检查 `CORS_ORIGINS`。
