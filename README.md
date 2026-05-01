# MultiAgents System For Career Planning

基于 BeMyEyes 思路的多模态职业规划助手：
- 本地小模型负责多模态感知（文本/图像/文档/音频）
- 云端 DeepSeek API 负责规划推理大脑
- 支持 CLI 与 API（含 SSE 流式）

## 核心模块
- `project/orchestrator.py`：统一编排（感知 -> 大脑 -> 回退）
- `project/agents/perception/`：本地感知代理（text/image/document/audio）
- `project/core/brain_client.py`：DeepSeek 大脑与本地回退大脑
- `project/core/schemas.py`：统一结构化契约
- `project/api/api.py`：FastAPI + SSE
- `project/main.py`：CLI 交互入口（支持流式）

## 依赖安装（agents 环境）
```bash
conda activate agents
pip install fastapi uvicorn httpx sse-starlette pydantic-settings pypdf
```

## 环境变量
PowerShell:
```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:BRAIN_DEFAULT_MODEL="deepseek-chat"
$env:BRAIN_TIMEOUT_SECONDS="45"
$env:BRAIN_RETRY_TIMES="2"
```

## CLI
非流式：
```bash
python -m project.main --session-id demo-1 --goal "我想在6个月内转岗数据分析" --text "我会Python和SQL，但项目经验不足" --city 上海 --time-budget 10
```

流式：
```bash
python -m project.main --stream --session-id demo-1 --goal "我想在6个月内转岗数据分析" --text "我会Python和SQL，但项目经验不足" --city 上海 --time-budget 10
```

## API
启动：
```bash
python -m project.api.run_api
```

PowerShell 中文调用建议：
```powershell
chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

普通接口：
```bash
curl -X POST "http://127.0.0.1:8000/v1/assist" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-1\",\"user_goal\":\"我想转岗数据分析\",\"text_input\":\"我会Python和SQL\"}"
```

SSE 流式接口：
- `POST /v1/assist/stream`
- 事件类型：`stage_start`、`stage_progress`、`token`、`stage_end`、`final_result`、`error`

## 回归测试
```bash
python -m unittest test_model.test_mvp_components
```

## PowerShell API 一键测试
```powershell
.\test_api.ps1 -SessionId "demo-utf8-1"
```

如需排查兼容性，可关闭“原始 UTF-8 解码模式”：
```powershell
.\test_api.ps1 -SessionId "demo-utf8-1" -UseRawUtf8:$false
```
