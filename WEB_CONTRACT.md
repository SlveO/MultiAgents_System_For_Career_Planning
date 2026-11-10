# 第一阶段 Web 合同

本地 Demo 补充：`scripts/web/start-local-demo.sh` 启动真实 API 与网页，网页默认127.0.0.1:5180、API默认127.0.0.1:8788。仅显式开发模式使用固定 `career_local_demo` 身份，数据独立于原 Web/CLI；浏览器同源 `/api` 请求复用代理合同转成 `/v1`，不经过 Sites 或隧道。身份和服务令牌由本机服务补充，不能由浏览器指定。生产构建和普通开发模式不启用本地身份。

本次方案授权新增 Web/API；CLI 继续使用现有编排。禁止训练、下载权重或 CPU 模型回退。工作区已有修改必须保留。本次不提交、推送或合并 Git。

## HTTP

浏览器只访问 Sites `/api/...`；服务端代理到 GPU `/v1/...`。后端要求 `Authorization: Bearer <CAREER_SERVICE_TOKEN>` 和 `X-Career-User: <Sites stable user ID>`，不信任浏览器传来的身份。所有写请求必须有 `Idempotency-Key`；相同用户/键/请求重放原结果，不同请求返回 409。错误为 `{"error":{"code":"...","message":"..."}}`。

- `GET /v1/sessions` → `{sessions: SessionView[]}`
- `POST /v1/sessions`，`{title?, user_goal, text_input?, text_subject?: student|job|reference|unknown, output_preferences?}` → SessionView；文本主体默认 unknown
- `GET /v1/sessions/{id}` → SessionView（记录恢复/查看事件）
- `POST /v1/sessions/{id}/uploads`，multipart `file`、`subject`（student/job/reference/unknown，默认 unknown） → Upload
- `POST /v1/sessions/{id}/perceive`，`{state_version}` → Task
- `POST /v1/sessions/{id}/answers`，`{state_version, question_id, answer}` → Task
- `POST /v1/sessions/{id}/plans`，`{state_version}` → Task
- `POST /v1/sessions/{id}/feedback`，`{state_version, plan_id, source_version, feedback}` → Task，feedback 为 过短/合适/过于详细
- `POST /v1/sessions/{id}/evaluations`，`{plan_id, output_version, scores?: {usefulness?: 1..5, clarity?: 1..5, feasibility?: 1..5}, comment?: string}` → 保存的评价；无默认评分
- `POST /v1/sessions/{id}/events`，`{event: view|restore|graph_view, plan_id?, output_version?}` → `{ok:true}`
- `GET /v1/sessions/{id}/versions` → `{versions: OutputVersion[]}`
- `GET /v1/sessions/{id}/export?format=json|markdown` → 用户导出，使用已存快照
- `DELETE /v1/sessions/{id}` → `{deleted:true}`，撤销访问、取消排队、丢弃运行结果
- `GET /v1/tasks/{id}` → Task
- `GET /healthz` → 不含身份或密钥的健康结果

SessionView: `{id,title,user_goal,text_input,status,state_version,question,uploads,active_task,plan,versions,created_at,updated_at}`。可附加 `source_claims`、`knowledge_snapshot`、`evaluations`。不暴露内部文件路径。question 为 null 或 `{id,field,prompt,reason}`。plan 为原始 CareerPlanResponse JSON（可附 graph、report_markdown、knowledge_snapshot），versions 为 OutputVersion 数组，以最后一项呈现。

Upload: `{id,name,size,subject,created_at,deleted_at?}`。每会话最多3个，每个25 MiB，支持 txt/md/csv/tsv/pdf/docx/xlsx/png/jpg/jpeg/webp/mp4/avi/mov/mkv，不支持音频。

Task: `{id,session_id,kind,status,error?,created_at,updated_at}`。status 为 queued/running/succeeded/failed/interrupted/cancelled。单进程单 worker、用户最多一个 queued/running，全局至多10个 queued。状态冲突 409，容量 429，隔离对象不存在 404，存储故障 503。写操作原子化；无自动模型/API重试。运行中重启标 interrupted，queued 恢复。

## Python 引擎（主 Agent）

`project.core.web_workflow.WebPlanningEngine.process(kind, state, payload, files) -> dict`，同步，由唯一后台线程调用；kind 为 perceive/answers/plans/feedback。state 是上次返回的完整 JSON 字典，首次为创建会话数据（含 id,user_goal,text_input,output_preferences）。files 为 `{id,path,name,subject}` 数组，path 只在服务端。

返回完整新 state，包含 `status,question,request,profile,perception_results,rounds,plan,versions,source_claims,knowledge_snapshot,extracted`；status 为 collecting/awaiting_answer/ready/planned。引擎不写 CLI SQLite/JSONL。后端仅在会话仍有效时原子提交新 state 并增加 state_version。extracted=true 后原文件立即删除。模型失败必须让任务明确失败，保留原 state，不伪装成功。

## 知识与来源（Agent C）

`KnowledgeRegistry(root=..., seed_path=...).snapshot(query, top_k=4)` 返回 `{version_id,content_sha256,hits,references}`。hits 为完整列表字段条目，references 与 CareerKnowledgeBase.retrieve 格式一致。`current()` 返回版本记录，`list_versions()` 列表。初始化固定65条 ID；审核/发布/回滚仅服务器命令。报告和图谱只消费保存的 snapshot，不回读现行库。

来源协议须区分字段 subject=student/job/reference/unknown 与 verification=unverified/user_confirmed/source_verified，不能把岗位要求推成学生技能，也不能把用户确认升级成外部核验。主 Agent 接入感知和引导；Agent C 提供验证合同与纯函数。

## 分工

- 主 Agent：统筹计划、调度、共有接口、整合、工作记录汇总和最终结论。
- `gpt-6-luna max`：大量读取、只读定向扫描，以及文档、配置和简单写入修复。
- `gpt-6-sol xhigh`：仅处理实际出现的跨模块推理、复杂功能或代理集成，以及疑难修复。

编号顺序、状态和阶段边界见 [DEMO_PLAN.md](DEMO_PLAN.md)；实际执行结果与交接见 [AGENT_WORKLOG.md](AGENT_WORKLOG.md)。

原文件提取后立即删、最迟24小时；会话/规划/评价/事件30天；清理器定时与管理命令均可运行。删除无恢复入口。项目模型调用仅在 `DEMO_PLAN` 第 3 项按当前预算执行，第 1、2 项不调用；不得伪造人工分数或私有网址。
