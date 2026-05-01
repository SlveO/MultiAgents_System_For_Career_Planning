# 项目技术协作文档

## 项目概述
基于 BeMyEyes 思路的多模态职业规划助手。本地小模型（Qwen3-VL / Whisper）负责将图片、文档、音频等非文本模态翻译为文本描述，云端 DeepSeek API 负责所有文本推理与规划生成，支持 CLI 与 API（含 SSE 流式）。

## 项目最终目标
- [x] 多模态输入分类与路由（text/image/multimodal/file/audio_video）
- [x] 本地小模型感知（Qwen3-VL-2B-Instruct 视觉, Whisper-small 语音, 规则文本分析）
- [x] 文本类数据直接送云端 DeepSeek API（不走本地模型）
- [x] 云端 DeepSeek API 推理 + 规则模板回退
- [x] FastAPI + SSE 流式 API
- [x] RAG 职业知识库 — 向量嵌入 + 混合检索（关键词 + 语义）+ 21 职业角色
- [x] SQLite 会话持久化
- [x] 三套并行系统统一为一致架构
- [x] 移除本地文本生成模型（DeepSeek-R1-Distill-Qwen-1.5B）
- [x] RAG 升级为向量嵌入 + 混合检索 + 重排序
- [x] 知识库扩充（21 个职业角色）
- [x] 视频关键帧提取（cv2 + Qwen3-VL 帧描述）
- [x] 多模态感知结果统一为 PerceptionResult 结构（含 MultiModalFusion 融合）
- [x] Web 前端 + 用户认证（JWT）+ Docker 部署 + 文件上传

## 当前阶段目标

**优先级 1（安全 + 一致性 + 架构修正）— 已完成 2026-05-02：**
- [x] 移除硬编码 API Key
- [x] 修复 test_mvp_components.py 导入路径
- [x] 修复 README.md 过期路径
- [x] 三套并行系统统一
- [x] 移除本地 1.5B 文本模型
- [x] TextPerceptionAgent 改为规则模式
- [x] 删除 LocalBrainFallbackClient 死代码
- [x] DeepSeekBrainClient.plan_stream() 边界增强

**优先级 2（RAG 升级）— 已完成 2026-05-02：**
- [x] sentence-transformers + bge-small-zh-v1.5（24MB，ModelScope 下载）文本向量化
- [x] ChromaDB 向量数据库（持久化存储到 ./data/chroma_db）
- [x] 混合检索：关键词加权(0.3) + 向量语义(0.7) + 融合排序
- [x] 知识库扩充：5 → 21 个职业角色

**优先级 3（多模态完善）— 已完成 2026-05-02：**
- [x] 3.1 Whisper-small 模型下载（ModelScope openai-mirror/whisper-small → ./models/）
- [x] 3.2 AudioAgent 可配置路径 + 设备自动检测（CUDA→CPU 降级）
- [x] 3.3 DocumentAgent 扩展 .docx / .xlsx / .csv / .tsv 支持
- [x] 3.4 VideoPerceptionAgent 创建（cv2 关键帧提取 + Qwen3-VL 帧描述）
- [x] 3.5 所有感知代理置信度标准化（使用 base._safe_confidence()）
- [x] 3.6 MultiModalFusion 重写（PerceptionResult 输入，置信度加权排序，跨模态去重）
- [x] 3.7 MultimodalChatPipeline 重构（使用 PerceptionAgent 替代直接处理，添加 lazy init）
- [x] 3.8 全部 17 个测试通过

**优先级 4（产品化）— 已完成 2026-05-02：**
- [x] 4.1 Dockerfile + docker-compose.yml + .dockerignore + requirements-docker.txt
- [x] 4.2 CORS 中间件 + api_host/api_port/cors_origins settings
- [x] 4.3 JWT 用户认证（/auth/register + /auth/login + get_current_user 依赖注入）
- [x] 4.4 Web 前端（单页 HTML：登录、职业规划 SSE 流式、对话 SSE 流式、文件上传）
- [x] 4.5 文件上传端点（POST /v1/upload）+ StaticFiles 挂载
- [x] 4.6 memory_manager.py GBK 编码修复（所有 emoji/中文 print 替换为 ASCII）

## 正确的数据流
```
文本输入 ─────────────────────────────► 云端 DeepSeek API ──► 流式输出
                                          ↑
图片 ──► Qwen3-VL ──► 文本描述 ──────────┤  (作为上下文注入)
文档 ──► 解析器  ───► 文本提取 ──────────┤
音频 ──► Whisper ───► ASR文本 ───────────┤
视频 ──► cv2关键帧 ─► Qwen3-VL帧描述 ────┤
                                          │
RAG检索 ──► 关键词 + 向量混合检索 ────────┘  (上下文知识注入)
                                          │
MultiModalFusion ──► 置信度加权+去重 ─────┘  (多模态融合)
```

## 优先级 3 关键变更
| 变更 | 说明 |
|------|------|
| 下载 Whisper-small | ModelScope openai-mirror/whisper-small，922MB，16 文件 |
| AudioAgent 重构 | 可配置 `model_path` + `device="auto"`，CUDA 不可用时自动 CPU |
| DocumentAgent 扩展 | 支持 .docx (python-docx) / .xlsx (openpyxl) / .csv / .tsv |
| 新建 VideoPerceptionAgent | cv2 关键帧采样（间隔 5s，最多 5 帧）+ Qwen3-VL 帧描述 |
| Schemas 扩展 | ModalityType 添加 "video"，TaskRequest 添加 `video_paths` |
| 置信度标准化 | 全部 agent 使用 `_safe_confidence()`：text 0.55-0.80，image 0.45-0.65，document/audio 0.4-0.6 |
| MultiModalFusion 重写 | `fuse(results: List[PerceptionResult]) -> str` 置信度加权 + 跨模态去重 |
| Pipeline 重构 | `_understand_images()` 使用 ImagePerceptionAgent；`_understand_documents()` 使用 DocumentPerceptionAgent；`_understand_audio_video()` 使用 VideoPerceptionAgent |
| Orchestrator 重构 | 添加 VideoPerceptionAgent + MultiModalFusion.fuse() 替换原生字符串拼接 |
| memory_manager GBK 修复 | 全部 print 语句从 emoji/中文改为 ASCII（避免 Windows GBK 编码错误） |

## 优先级 4 关键变更
| 变更 | 说明 |
|------|------|
| Dockerfile | python:3.11-slim，CPU-only API（GPU 模型在宿主机），端口 8000 |
| docker-compose.yml | API 服务 + 环境变量注入 + 数据卷挂载 |
| .dockerignore | 排除 models/、__pycache__/、.git/、测试临时文件 |
| requirements-docker.txt | 精简依赖（fastapi, uvicorn, httpx, chromadb, sentence-transformers, sse-starlette, pypdf, python-docx, openpyxl, python-multipart, python-jose, passlib） |
| CORS 中间件 | allow_origins 从 settings.cors_origins 读取，默认 "*" |
| JWT 认证 | bcrypt 密码哈希 + JWT token（python-jose），`get_current_user` 依赖注入保护 /v1/* 端点 |
| /auth/register | 注册返回 user_id + api_key |
| /auth/login | 登录返回 access_token + token_type |
| /v1/upload | 多部分文件上传，自动检测类型（image/document/audio/video），返回文件路径 |
| Web 前端 | project/static/index.html — 单页应用：登录、职业规划（SSE 流式）、对话助手（SSE 流式）、文件上传、会话历史、暗色主题响应式 |
| settings 扩展 | api_host, api_port, cors_origins, jwt_secret_key, jwt_algorithm, jwt_expire_minutes |

## 已完成模块
- **输入路由** — [project/core/input_router.py](project/core/input_router.py)：InputClassifier（5 种模态分类）+ DataRouter
- **多模态管道** — [project/core/multimodal_pipeline.py](project/core/multimodal_pipeline.py)：routing→small_model→rag→llm_stream→final
- **职业编排器** — [project/orchestrator.py](project/orchestrator.py)：意图检测 + 感知采集 + 画像构建 + 云端规划 + 规则模板回退
- **CLI 助手** — [project/main.py](project/main.py)：全部 5 种模态，文本走云端流式
- **FastAPI 服务** — [project/api/api.py](project/api/api.py)：16 个路由（含 auth + CORS + 文件上传 + 静态页面）
- **大脑客户端** — [project/core/brain_client.py](project/core/brain_client.py)：仅 DeepSeekBrainClient（云端 SSE 流式）
- **职业知识库** — [project/core/career_knowledge.py](project/core/career_knowledge.py)：ChromaDB 向量存储 + 关键词混合检索（bge-small-zh-v1.5），21 个职业角色
- **会话记忆** — [project/core/session_memory.py](project/core/session_memory.py)：SQLite + JSON
- **显存管理** — [project/core/memory_manager.py](project/core/memory_manager.py)：只管理 vision 模型，ASCII 日志
- **用户认证** — [project/core/auth.py](project/core/auth.py)：JWT + bcrypt + SQLite，register/login/get_current_user
- **感知代理** — [project/agents/perception/](project/agents/perception/)：text（规则）/ image（Qwen-VL）/ document（含 docx/xlsx）/ audio（Whisper）/ video（cv2+Qwen-VL）
- **图像处理器** — [project/agents/image.py](project/agents/image.py)：Qwen3-VL-2B-Instruct
- **多模态融合** — [project/utils/fusion.py](project/utils/fusion.py)：MultiModalFusion（PerceptionResult 置信度加权 + 跨模态去重）
- **数据契约** — [project/core/schemas.py](project/core/schemas.py)：Pydantic 模型（含 video 模态 + video_paths）
- **Web 前端** — [project/static/index.html](project/static/index.html)：单页应用（登录、规划 SSE、对话 SSE、文件上传）

## 代码架构概览

```
project/
├── main.py                  # CLI 入口（全部 5 种模态，文本走云端）
├── orchestrator.py          # 职业规划编排器（云端优先 + 规则回退）
├── core/
│   ├── auth.py              # JWT 用户认证（register/login/get_current_user）
│   ├── schemas.py           # Pydantic 数据模型
│   ├── settings.py          # 配置（API Key / JWT / CORS / 网络）
│   ├── input_router.py      # 输入分类与路由
│   ├── multimodal_pipeline.py # 标准多模态管道
│   ├── brain_client.py      # DeepSeek 云端客户端
│   ├── career_knowledge.py  # 职业知识库（ChromaDB + 混合检索）
│   ├── session_memory.py    # SQLite 会话持久化
│   └── memory_manager.py    # GPU 显存管理（仅 vision）
├── agents/
│   ├── image.py             # Qwen3-VL 图像处理器
│   └── perception/          # 感知代理
│       ├── base.py          # 公共工具函数
│       ├── text_agent.py    # 规则文本分析（无模型）
│       ├── image_agent.py   # 图像感知
│       ├── document_agent.py # 文档感知（txt/md/csv/pdf/docx/xlsx）
│       ├── audio_agent.py   # 音频感知（Whisper-small）
│       └── video_agent.py   # 视频关键帧提取与描述
├── api/
│   ├── api.py               # FastAPI 应用（16 路由 + CORS + 认证）
│   └── run_api.py           # API 启动脚本
├── utils/
│   └── fusion.py            # 多模态融合（MultiModalFusion）
├── static/
│   └── index.html           # Web 前端（单页应用）
└── tests/
    ├── test_input_router.py
    ├── test_multimodal_pipeline.py
    ├── test_multimodal_api_flow.py
    └── test_main_entry_ast.py
```

## 关键文件索引
- [project/core/settings.py](project/core/settings.py) — 全局配置（API Key / JWT / CORS / 网络）
- [project/core/schemas.py](project/core/schemas.py) — 数据契约（含 video 模态 + video_paths）
- [project/core/input_router.py](project/core/input_router.py) — 输入分类与路由
- [project/core/multimodal_pipeline.py](project/core/multimodal_pipeline.py) — 标准管道（含 agent 注入支持）
- [project/core/brain_client.py](project/core/brain_client.py) — DeepSeek API 客户端（SSE）
- [project/core/career_knowledge.py](project/core/career_knowledge.py) — 职业知识库（ChromaDB + 混合检索）
- [project/core/auth.py](project/core/auth.py) — JWT 用户认证系统
- [project/orchestrator.py](project/orchestrator.py) — 职业规划编排（含 MultiModalFusion + VideoAgent）
- [project/main.py](project/main.py) — CLI 交互入口（全部模态）
- [project/api/api.py](project/api/api.py) — FastAPI 端点（16 路由 + 认证 + CORS + 上传）
- [project/agents/perception/](project/agents/perception/) — 全部 5 种感知代理
- [project/agents/perception/video_agent.py](project/agents/perception/video_agent.py) — 视频关键帧提取代理
- [project/utils/fusion.py](project/utils/fusion.py) — 多模态融合处理器
- [project/static/index.html](project/static/index.html) — Web 前端
- [test_model/test_mvp_components.py](test_model/test_mvp_components.py) — MVP 测试
- [dataset/career_knowledge_base.json](dataset/career_knowledge_base.json) — 职业知识库 JSON（21 条）
- [data/chroma_db/](data/chroma_db/) — ChromaDB 向量持久化目录
- [data/auth.db](data/auth.db) — 用户认证数据库（SQLite）
- [Dockerfile](Dockerfile) — Docker 镜像构建
- [docker-compose.yml](docker-compose.yml) — 容器编排
- [requirements-docker.txt](requirements-docker.txt) — Docker 精简依赖

## 当前状态与进度

**已完成（截至 2026-05-02）：**
- 优先级 1-4 全部完成
- 17/17 单元测试通过
- Web 前端核心功能可用：登录注册、职业规划（SSE 流式）、对话助手（SSE 流式）、文件上传
- DeepSeek API 云端推理正常（需配置 DEEPSEEK_API_KEY）

**待完善：**
- 历史对话管理功能
- Web 前端 XSS 防护
- 生产环境 JWT 密钥配置

## 已知问题
1. **GPU 模型在 Docker 不可用**：Docker 镜像是 CPU-only，Qwen3-VL / Whisper 需在宿主机运行
2. **Whisper 模型较大**：922MB，首次下载耗时较长
3. **视频处理依赖 cv2**：需 opencv-python，已在 requirements 中
4. **JWT 密钥默认值不安全**：生产环境需通过 JWT_SECRET_KEY 环境变量覆盖
5. **前端无 XSS 防护**：单页 HTML 直接操作 innerHTML，仅用于开发/演示
6. **DeepSeek API Key 必须配置**：不配置时对话助手不返回内容，职业规划退回固定模板

---
> 最后更新：2026-05-02  by Agent (Claude Code)
