# MultiAgents System For Career Planning

基于多智能体的多模态职业规划助手：
- 本地小模型负责多模态感知（文本/图像/文档/音频/视频 → 文本描述）
- 云端 DeepSeek API 负责规划推理大脑
- 支持 CLI、REST API（含 SSE 流式）、Web 前端

## 架构

```
文本输入 ────────────────────────────► DeepSeek API ──► 流式输出
                                         ↑
图片 ──► Qwen3-VL    ──► 文本描述 ───────┤
文档 ──► 解析器       ──► 文本提取 ───────┤
音频 ──► Whisper     ──► ASR 文本 ───────┤
视频 ──► cv2 + Qwen  ──► 帧描述 ─────────┤
                                         │
RAG  ──► ChromaDB + bge-small-zh-v1.5 ───┘  (混合检索)
                                         │
MultiModalFusion ──► 置信度加权 + 去重 ───┘
```

## 项目结构

```
project/
├── main.py                     # CLI 交互入口（5 种模态）
├── orchestrator.py             # 职业规划编排器（云端优先 + 规则回退）
├── core/
│   ├── auth.py                 # JWT 用户认证（register/login）
│   ├── brain_client.py         # DeepSeek API 客户端（SSE 流式）
│   ├── career_knowledge.py     # 职业知识库（ChromaDB + 混合检索，21 角色）
│   ├── input_router.py         # 输入分类（5 模态）+ 数据路由
│   ├── memory_manager.py       # GPU 显存管理（vision 模型）
│   ├── multimodal_pipeline.py  # 多模态对话管道（route→小模型→RAG→LLM）
│   ├── schemas.py              # Pydantic 数据契约
│   ├── session_memory.py       # SQLite 会话持久化
│   └── settings.py             # 全局配置（环境变量）
├── agents/
│   ├── image.py                # Qwen3-VL 图像处理器
│   └── perception/             # 感知代理
│       ├── base.py             # 公共工具（置信度标准化）
│       ├── text_agent.py       # 规则文本分析（无模型）
│       ├── image_agent.py      # 图像感知（Qwen3-VL）
│       ├── document_agent.py   # 文档感知（txt/md/csv/pdf/docx/xlsx）
│       ├── audio_agent.py      # 音频感知（Whisper-small）
│       └── video_agent.py      # 视频关键帧提取 + Qwen3-VL 描述
├── api/
│   ├── api.py                  # FastAPI 应用（16 路由 + CORS + 认证 + 上传）
│   ├── run_api.py              # API 启动入口
│   ├── chat_from_file.py       # 文件批量调用工具
│   └── request.sample.json     # 请求样例
├── utils/
│   └── fusion.py               # MultiModalFusion（跨模态融合 + 去重）
├── static/
│   └── index.html              # Web 前端（单页应用）
└── tests/                      # 单元测试（14 个）
```

## 快速开始

### 1. 环境配置

```bash
conda activate agents
pip install -r requirements-docker.txt
```

GPU 模型额外依赖（需要 CUDA）：
```bash
pip install torch torchvision opencv-python modelscope
```

### 2. 设置 API Key

在项目根目录创建 `.env` 文件（已自动加入 .gitignore）：

```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

支持的环境变量（有默认值）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `BRAIN_DEFAULT_MODEL` | `deepseek-chat` | 默认模型 |
| `BRAIN_TIMEOUT_SECONDS` | `45` | 超时秒数 |
| `BRAIN_RETRY_TIMES` | `2` | 重试次数 |
| `JWT_SECRET_KEY` | `change-me-in-production` | JWT 签名密钥 |
| `CORS_ORIGINS` | `*` | CORS 允许域名 |
| `API_HOST` | `0.0.0.0` | 监听地址 |
| `API_PORT` | `8000` | 监听端口 |

### 3. 下载本地模型（可选，GPU 模式下需要）

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', cache_dir='./models')"
python -c "from modelscope import snapshot_download; snapshot_download('openai-mirror/whisper-small', cache_dir='./models')"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./models')"
```

## CLI 使用

```bash
python -m project.main --session-id demo-1 --goal "我想在6个月内转岗数据分析" --text "我会Python和SQL" --city 上海 --time-budget 10
```

流式输出：
```bash
python -m project.main --stream --session-id demo-1 --goal "转岗数据分析" --text "会Python和SQL"
```

## API 使用

### 启动服务

```bash
python -m project.api.run_api
```

### 端点一览

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| GET | `/healthz` | 无 | 健康检查 |
| POST | `/auth/register` | 无 | 注册（username + password） |
| POST | `/auth/login` | 无 | 登录（返回 JWT token） |
| POST | `/v1/assist` | JWT | 职业规划（非流式） |
| POST | `/v1/assist/stream` | JWT | 职业规划（SSE 流式） |
| GET | `/v1/session/{id}` | JWT | 获取会话历史和档案 |
| POST | `/v1/feedback` | JWT | 提交反馈 |
| POST | `/v1/upload` | JWT | 上传文件（自动识别类型） |
| POST | `/v1/multimodal/chat/stream` | 可选 | 通用对话（SSE 流式） |
| GET | `/v1/multimodal/chat/session/{id}` | 无 | 获取对话历史 |
| DELETE | `/v1/multimodal/chat/session/{id}` | 无 | 清除对话历史 |
| GET | `/` | 无 | Web 前端页面 |

### 调用示例

注册并登录：
```bash
curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo1234"}'

curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo1234"}'
```

职业规划（需 Bearer token）：
```bash
curl -X POST http://localhost:8000/v1/assist \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"session_id":"s1","user_goal":"转行数据分析","text_input":"会Python和SQL"}'
```

流式对话（可选认证）：
```bash
curl -X POST http://localhost:8000/v1/multimodal/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_input":"我适合做什么工作？"}'
```

## Web 前端

服务启动后浏览器打开 `http://localhost:8000`，提供：
- 用户注册/登录
- 职业规划（表单 + 流式 SSE）
- 对话助手（SSE 流式）
- 文件上传（自动类型检测）
- 暗色主题、响应式布局

## Docker 部署

```bash
docker build -t career-assistant .
docker run -p 8000:8000 -e DEEPSEEK_API_KEY=sk-xxx career-assistant
```

或使用 docker-compose：
```bash
DEEPSEEK_API_KEY=sk-xxx docker-compose up -d
```

注意：Docker 镜像为 CPU-only，不包含 GPU 推理模型（Qwen3-VL / Whisper）。图像和音频理解功能需在宿主机运行。

## 测试

```bash
# 全部测试（17 个）
python -m unittest discover -s project/tests -v
python -m unittest test_model.test_mvp_components -v
```

## 依赖一览

核心：`fastapi` `uvicorn` `httpx` `sse-starlette` `pydantic` `pydantic-settings`

RAG：`chromadb` `sentence-transformers`

文档解析：`pypdf` `python-docx` `openpyxl`

认证：`python-jose[cryptography]` `passlib[bcrypt]` `python-multipart`

GPU（本地模型）：`torch` `torchvision` `opencv-python` `modelscope`
