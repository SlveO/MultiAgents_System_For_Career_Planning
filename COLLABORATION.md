# 项目技术协作文档

> **Codex 结题版状态（2026-08-14）**：本文后续内容保留 Claude 阶段的架构与前端历史，便于追溯，不再作为当前运行说明。当前结题 MVP 以 [README.md](README.md)、[AGENTS.md](AGENTS.md) 和 `docs/superpowers/plans/2026-08-14-completion-mvp.md` 为准。唯一 CLI 是 `python -m project.assistant_cli`；默认模型为 `deepseek-v4-flash` 且关闭思考模式；关键词知识库为 50 条；文本与受支持文档无需 GPU；图像、音频、视频、向量检索、API 与 Web 均为可选能力。当前新增流程包含 8 个固定追问、三档反馈、隐私脱敏 JSONL 日志及四组可重复实验。

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
- [x] Web 前端现代化重构（React 18 + Vite 5 + TypeScript 5 + Tailwind CSS 3）— 2026-05-02
- [x] 前后端一键启动（Vite 插件自动 spawn 后端）— 2026-05-02
- [x] SSE 流式超时保护（15s 连接 + 30s 流读取）+ 管道进度展示 — 2026-05-02
- [x] 对话助手 / 职业规划功能区分 + 侧边栏折叠修复 — 2026-05-02
- [x] 会话历史字段格式对齐（后端 {user,assistant} ↔ 前端 {role,content}）— 2026-05-02
- [x] 文件上传代理绕过（multipart 直连后端）— 2026-05-02
- [x] 会话切换本地数据保护（防止后端覆盖失败对话）— 2026-05-02
- [x] 文件上传错误详情增强（HTTP 状态码 + 响应体解析）— 2026-05-02
- [x] 文件路径检测修复（消息格式去除路径括号）— 2026-05-02
- [x] 对话助手会话历史 SQLite 持久化（后端重启不丢失）— 2026-05-03
- [x] JWT 密钥启动时自动生成（检测到默认值随机生成并警告）— 2026-05-03

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

**Web 前端现代化重构 — 2026-05-02：**
- [x] 技术栈升级：原生 HTML/CSS/JS → React 18 + Vite 5 + TypeScript 5 + Tailwind CSS 3
- [x] 后端自动启动：Vite 插件 `spawnBackend` 通过 `conda run` 一键启动前后端
- [x] SSE 流式超时保护：15s 连接超时 + 30s 流读取超时 + 明确错误提示
- [x] 管道进度展示：5 阶段进度提示（路由→分析→RAG→提示词→生成）
- [x] SSE 解析器 \r\n 归一化修复（HTTP CRLF 事件边界兼容）
- [x] 会话历史字段对齐：后端 `{user,assistant}` → 前端 `{role,content}` 转换
- [x] 对话助手 / 职业规划功能区分：导航栏"对话助手"tab + 右侧"职业规划提问"面板
- [x] 侧边栏折叠修复：组件始终渲染，折叠态显示展开按钮
- [x] 待办持久化：localStorage（原版刷新丢失数据）
- [x] 技术文档更新：技术文档.md V2.0（反映 React 架构）
- [x] 旧文件清理：page1.html / style.css / script.js（替换为 38 个模块化源文件）
- [x] 文件上传修复：multipart 直连后端绕过 Vite 代理（60s 超时 + JWT 手动注入）
- [x] 文件上传错误详情：HTTP 状态码 + 响应体 `detail`/`message` 字段解析
- [x] 文件路径检测修复：消息格式 `[附件: file]\npath` 确保后端正则 `^[A-Za-z]:` 可匹配
- [x] 会话本地数据保护：已有用户消息的会话不被后端数据覆盖（保留失败对话记录）
- [x] 会话历史 SQLite 持久化：MultimodalPipeline 注入 SessionMemory，对话历史写入 SQLite
- [x] JWT 密钥安全启动：检测默认值自动生成 os.urandom(32) 随机密钥

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
project/                       # 后端 Python 项目
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
│       ├── base.py
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
│   └── index.html           # Web 前端（旧版，已被 web/ 取代）
└── tests/
    ├── test_input_router.py
    ├── test_multimodal_pipeline.py
    ├── test_multimodal_api_flow.py
    └── test_main_entry_ast.py

web/                           # 前端 React 项目（现代化重构）
├── index.html               # Vite 入口 HTML
├── package.json             # 依赖管理（react, vite, tailwindcss, lucide-react）
├── vite.config.ts           # Vite 配置 + spawnBackend 插件 + 代理
├── tailwind.config.ts       # Tailwind 自定义主题色
├── tsconfig.json            # TypeScript 项目引用
├── postcss.config.js        # PostCSS 配置
└── src/
    ├── main.tsx             # React 入口（挂载 Providers + App）
    ├── App.tsx              # 根组件：认证门控 + 布局 + 模态窗协调
    ├── index.css            # Tailwind 指令 + 全局样式 + 翻页时钟 CSS
    ├── api/                 # API 请求层
    │   ├── client.ts        # fetch 封装（JWT 注入、超时、SSE 流读取）
    │   ├── auth.ts          # login(), register(), registerAndLogin()
    │   ├── chat.ts          # sendChatMessage() 流式生成器 + 阶段进度
    │   └── upload.ts        # uploadFile() multipart
    ├── store/               # 状态管理（React Context）
    │   ├── AuthContext.tsx    # 用户认证、JWT 持久化
    │   ├── ChatContext.tsx    # 会话管理、SSE 流式消息
    │   ├── TodoContext.tsx    # 待办 CRUD、拖拽排序、localStorage
    │   └── PomodoroContext.tsx# 番茄钟状态机
    ├── components/          # React 组件（20 个）
    │   ├── layout/          # Header, Navbar
    │   ├── auth/            # LoginForm, RegisterForm
    │   ├── chat/            # ChatContainer, MessageBubble, ChatInput,
    │   │                    #   TypingIndicator, QuickQuestions, FileAttachment
    │   ├── history/         # HistorySidebar（可折叠，始终渲染）
    │   ├── modals/          # Modal, InfoModal, TodoDetailModal,
    │   │                    #   PomodoroModal, FlipClockModal
    │   ├── todo/            # TodoForm, TodoList, TodoItem
    │   └── flipclock/       # FlipClock, FlipCard
    ├── hooks/
    │   └── useFlipClock.ts  # 翻页时钟定时器
    ├── types/
    │   └── index.ts         # TypeScript 类型定义（API + App State）
    └── utils/
        ├── format.ts        # 时间/日期/文本格式化
        └── storage.ts       # localStorage 安全读写
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
- [project/static/index.html](project/static/index.html) — Web 前端（旧版，已被 web/ 取代）
- [web/src/App.tsx](web/src/App.tsx) — React 根组件
- [web/src/api/client.ts](web/src/api/client.ts) — fetch 封装 + SSE 流读取（\r\n 归一化 + 超时保护）
- [web/src/api/chat.ts](web/src/api/chat.ts) — SSE 流式聊天调用 + 管道进度事件
- [web/src/store/ChatContext.tsx](web/src/store/ChatContext.tsx) — 会话状态管理（含后端历史格式转换）
- [web/src/store/AuthContext.tsx](web/src/store/AuthContext.tsx) — JWT 认证状态
- [web/src/components/chat/ChatInput.tsx](web/src/components/chat/ChatInput.tsx) — 消息输入组件
- [web/src/components/history/HistorySidebar.tsx](web/src/components/history/HistorySidebar.tsx) — 可折叠历史侧边栏
- [web/src/components/chat/QuickQuestions.tsx](web/src/components/chat/QuickQuestions.tsx) — 职业规划提问面板
- [web/vite.config.ts](web/vite.config.ts) — Vite 配置 + spawnBackend 后端自动启动插件
- [web/技术文档.md](web/技术文档.md) — 前端技术文档 V2.0
- [test_model/test_mvp_components.py](test_model/test_mvp_components.py) — MVP 测试
- [dataset/career_knowledge_base.json](dataset/career_knowledge_base.json) — 职业知识库 JSON（21 条）
- [data/chroma_db/](data/chroma_db/) — ChromaDB 向量持久化目录
- [data/auth.db](data/auth.db) — 用户认证数据库（SQLite）
- [Dockerfile](Dockerfile) — Docker 镜像构建
- [docker-compose.yml](docker-compose.yml) — 容器编排
- [requirements-docker.txt](requirements-docker.txt) — Docker 精简依赖

## 当前状态与进度

**已完成（截至 2026-05-03）：**
- 优先级 1-4 全部完成
- 14/14 单元测试通过（含 input_router 5, main AST 2, multimodal_pipeline 5, multimodal_api_flow 2）
- Web 前端 React 现代化重构完成（38 个模块化源文件，构建产物 191KB JS + 22KB CSS gzip ~59KB+5KB）
- 前端功能完整：JWT 登录/注册、对话助手（SSE 流式 + 5 阶段进度）、职业规划提问、待办管理（localStorage 持久化）、番茄钟、翻页时钟
- DeepSeek API 云端推理正常（需配置 DEEPSEEK_API_KEY）
- CLI 文件路径检测修复：`.match()` → `.search()` 支持中文文本中嵌入路径（无空格分隔）
- CLI 文件/音频模式始终送云端分析，即使无显式 text_context
- 前后端一键启动（`cd web && npm run dev`，Vite 插件自动 spawn 后端）
- SSE 解析器兼容 HTTP CRLF 换行符
- 会话历史前端 ↔ 后端字段格式对齐
- 文件上传全链路修复：Vite 代理绕过 + 60s 超时 + HTTP 错误详情 + 路径格式修复
- 会话数据保护：本地已有用户消息时不被后端数据覆盖
- 对话助手会话历史 SQLite 持久化：注入 SessionMemory 到 MultimodalPipeline
- JWT 密钥安全强化：启动时检测默认值自动生成随机密钥并警告

**待完善：**
- 生产环境 JWT 密钥应通过 `JWT_SECRET_KEY` 环境变量显式配置（启动时若检测到默认值会自动生成随机密钥并警告）

## 已知问题
1. **GPU 模型在 Docker 不可用**：Docker 镜像是 CPU-only，Qwen3-VL / Whisper 需在宿主机运行
2. **Whisper 模型较大**：922MB，首次下载耗时较长
3. **python-docx / openpyxl 为本地文档解析必需**：Docker 镜像已包含，本地 conda 环境需 `pip install python-docx openpyxl`
4. **conda 环境依赖**：Vite 插件使用 `conda run -n agents` 启动后端，需确保 conda 环境名匹配

## 前端重构关键变更记录（2026-05-02）

### 架构升级
| 旧版 | 新版 |
|------|------|
| 3 文件（HTML/CSS/JS ~2250 行） | 38 模块化 TypeScript 源文件 |
| Tailwind CSS CDN + Font Awesome CDN | Tailwind CSS 编译时 + lucide-react（tree-shakeable） |
| 无构建工具 | Vite 5（HMR + TypeScript + tree-shaking） |
| `sendMessageToAI()` mock 1s 延迟固定文本 | 真实 SSE 流式调用 `/v1/multimodal/chat/stream` |
| 待办数据刷新丢失 | localStorage 持久化 |
| 无认证 | JWT 登录/注册 |
| 死代码：9 个孤儿 DOM 引用 + 猴子补丁 | 零死代码 |

### Bug 修复记录
| 问题 | 根因 | 修复文件 |
|------|------|---------|
| 注册/登录失败 ECONNREFUSED | 后端未启动 | [vite.config.ts](web/vite.config.ts) — spawnBackend 插件 |
| 聊天永远显示"思考中"无返回 | SSE `reader.read()` 无超时 | [client.ts](web/src/api/client.ts) — 15s 连接 + 30s 流读取超时 |
| SSE 事件无法解析导致空返回 | HTTP CRLF `\r\n` vs parser `\n\n` 不匹配 | [client.ts](web/src/api/client.ts) — `.replace(/\r\n/g, '\n')` 归一化 |
| 切换会话后聊天记录消失 | 后端 `{user,assistant}` vs 前端 `{role,content}` 字段不匹配 | [ChatContext.tsx](web/src/store/ChatContext.tsx) + [types/index.ts](web/src/types/index.ts) |
| 侧边栏折叠后无法展开 | HistorySidebar 被条件渲染移除 | [App.tsx](web/src/App.tsx) — 始终渲染 `<HistorySidebar />` |
| 对话助手/职业规划界限不清 | tab 标签不准确 | [Navbar.tsx](web/src/components/layout/Navbar.tsx) — "职业规划"→"对话助手" + 右侧面板"职业规划提问" |
| 文件上传 "Upload failed" | Vite 代理破坏 multipart/form-data Content-Type | [upload.ts](web/src/api/upload.ts) — 直连 `http://localhost:8000/v1/upload` 绕过代理；60s 超时；手动注入 JWT |
| 上传错误信息不具体（仅 "Upload failed"） | 未解析 HTTP 响应体 | [upload.ts](web/src/api/upload.ts) — 解析 `res.json()` 中的 `detail`/`message` 字段；降级到 `res.text()` |
| 文件上传成功但后端报 "no valid document file found" | 前端消息格式 `[附件: file (D:\path.docx)]`，后端路径正则需要 `^[A-Za-z]:` 开头但匹配到 `(` | [ChatInput.tsx](web/src/components/chat/ChatInput.tsx) — 格式改为 `[附件: file]\nD:\path.docx`，路径独占一行无括号 |
| 失败对话在切换会话后消失 | 后端仅存储成功的 LLM 轮次；切换回会话时后端数据覆盖本地 | [ChatContext.tsx](web/src/store/ChatContext.tsx) — `hasLocalConversations` 守卫：本地有用户消息则跳过后端覆盖 |
| 对话助手聊天记录后端重启后丢失 | `MultimodalPipeline` 使用纯内存 dict 存储历史，未调用 `SessionMemory.append_interaction()` | [multimodal_pipeline.py](project/core/multimodal_pipeline.py) + [api.py](project/api/api.py) — 注入 `SessionMemory`；`run_stream()` 写入 SQLite；`get_session_history()` 内存缺失时从 DB 恢复 |
| JWT 密钥默认值 `change-me-in-production` 不安全 | 开发者常忘记设置环境变量 | [settings.py](project/core/settings.py) — `get_settings()` 检测默认值，自动用 `os.urandom(32).hex()` 生成随机密钥并 stderr 警告 |

---
> 最后更新：2026-05-03 by Agent (Claude Code) — 会话历史 SQLite 持久化 + JWT 密钥安全强化 + 协作文档清理
