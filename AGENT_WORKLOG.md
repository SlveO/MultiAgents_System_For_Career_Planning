# Demo 执行记录

本文件记录 `DEMO_PLAN.md` 的实际工作与当前状态。编号 1、2、3 仍按顺序推进；主 Agent 汇总已完成事项和必要检查。

## 阶段状态

| 项目 | 状态 | 依赖 / 剩余工作 |
|---|---|---|
| 计划编制 | 已审阅 | 无 |
| 1. 功能收尾 | 完成 | 复用前序文件精简成果；两路只读定向核对无确定主链路阻断缺陷，真实运行验证留在第 3 项 |
| 2. GitHub 源码交付 / 本机 Sites 交接 | 准备中 | 当前分支提交推送待主 Agent 实际执行；Sites 由本机桌面工作流后续创建和发布，服务器不直接发布 |
| 3. Demo 验证 | 已完成（据用户最新确认） | 本次交接不重做功能试跑或项目模型调用 |

## 当前交付状态（2026-11-10）

用户最新明确授权服务器将当前已验证工作区提交并推送到现有 GitHub 仓库的 `codex/proposal-expansion-20261105` 分支；不得强推、合并或推送 `main`，也不得更改仓库可见性。当前 GitHub 交付仍在准备中，实际提交和推送结果由主 Agent 完成后补记。本机 Sites 创建、同步和发布是后续步骤；服务器不直接发布 Sites。

此前删除的旧运行材料和兼容别名属于前序清理成果，不影响当前源码功能。本轮只同步已验证工作区；主 Agent 已完成干净暂存 web 源码副本构建，提交和推送仍待执行，不将其记为完成。功能验证已由用户确认完成，本轮不重跑，也不调用 GPU、DeepSeek 或项目 API。

## 计划与资料准备（历史）

| 时间 / 工作 | Agent / 模型 | 文件与结果 |
|---|---|---|
| 2026-11-10 06:40，计划初稿 | `/root/write_demo_plan_records`，`gpt-6-luna max` | 编写 `DEMO_PLAN.md`、`AGENT_WORKLOG.md`，在 `AGENTS.md` 增加计划入口，并刷新当时的 `SOURCE_MANIFEST.sha256`。明确桌面 Demo 范围、1→2→3 顺序与协作分工；没有启动编号任务、测试、构建、模型调用或部署。 |
| 编号 1 资料读取 | `/root/demo_function_plan_scan`，`gpt-6-luna max` | 只读查看申报功能和演示材料，未修改文件、未运行测试或调用项目模型，用于界定主链路缺项。 |
| 编号 2 资料读取 | `/root/demo_sites_plan_scan`，`gpt-6-luna max` | 只读查看 Sites、代理入口与现有条件，未部署或调用项目模型；当时未发现 Sites 原生创建/配置/发布工具、`hosting.json` 无 `project_id`、没有现成 cloudflared/HTTPS 入口，且暂停下载要求仍有效。 |
| 计划统筹、中文化和批注修订 | 主 Agent；文档修订 `/root/write_demo_plan_records` 与 `/root/revise_demo_review_comments`，均为 `gpt-6-luna max` | 修订 `DEMO_PLAN.md`、`AGENT_WORKLOG.md`、`AGENTS.md` 及当时的 `SOURCE_MANIFEST.sha256`；按批注明确直接移除过期要求及其专用实现、保留 Demo 共用主链路，并记录各路资料读取并非编号任务执行。未运行代码清理、测试、构建或部署，未调用项目模型。主 Agent 统筹模型未另行记录。 |
| 全项目清理方案与盘点 | `/root/write_full_cleanup_proposal`，`gpt-6-luna max`；依赖边界复核 `/root/cleanup_dependency_boundary_review`，`gpt-6-sol xhigh` | 当时方案写入 `DEMO_PLAN.md` 的“1.1 全项目精简方案”，并更新工作记录和清单。依据 3 项 Luna 盘点及 1 项 Sol 复核：106,293 个非软链文件、45 个软链、11,227 个目录；逻辑大小 7,105,199,697 B，`du` 为 7,262,855,168 B。方案阶段未执行删除。现方案小节已移除，实际结果记于下节。 |

## 本轮文件精简与收尾（2026-11-10）

### 文件精简执行

- **文档代理；`gpt-6-luna max`：** 删除 47 个文件、275,514 B 逻辑内容：`docs/` 32 份 Markdown、根目录 9 个旧文件、`design/` 3 个文件，以及 3 个子目录 README。将仍有效操作信息收敛到根 `README.md`、`AGENTS.md` 和父目录 `START-HERE.md`；Word 原件 238,454 B 未触碰，`WEB_CONTRACT.md` 保留。README 补入正式 CLI 调试帮助 `.venv-l20/bin/python -m project.assistant_cli --help`。本轮文档收尾由本 Agent 执行，实际模型 `gpt-6-luna max`，修改 `DEMO_PLAN.md`、`AGENT_WORKLOG.md`、`AGENTS.md` 和上述 README 单行；没有改源码或其他 README。
- **数据与缓存代理；`gpt-6-luna max`：** 删除 20,558 个常规文件、300,472,569 B 逻辑内容：Web 旧缓存/产物 159 个文件、15,126,681 B；旧数据目录 1,580 个、51,742,508 B；数据集旧样例 10 个、249,370 B；旧脚本 6 个、84,255 B；与 `.py` 成对的 `.pyc` 18,803 个、233,269,755 B。匿名 JSON 原样复制至 `dataset/demo/anonymous-export.json`（218,214 B），旧来源树已删除，因此逻辑删除统计不等于净空间回收。保留 `data/local-demo/`、`data/session_memory.db`、Node 22、现有环境、知识库、Word 原件和核心代码。
- **源码代理；`gpt-6-sol xhigh`：** 删除 54 个文件、192,567 B，包括旧兼容别名及其专属 AST 测试、completion protocol JSON 及唯一元数据测试方法、D1 模板、未用 CSS/connector 错误组件和 38 个无入口 UI 组件及 `use-mobile`；保留 23 个可达 UI 组件及功能 CLI/CUDA 测试。移除 16 项直接依赖；Node 22 离线同步锁文件成功且与 package 一致，`package-lock.json` 从 504,138 B 降至 427,722 B；离线 npm prune 成功，移除 70 个已安装包。未重装或下载；package import 已过滤不存在的 entries。
- 三组白名单直接删除共 **20,659 个文件 / 300,940,650 B 逻辑内容**，不含 npm 自动移除包内文件，也不代表实测净磁盘差值。数据组迁移的匿名样例仍保留。

### 主 Agent 已完成的必要检查

- Launcher `--check` 显示 ready；项目模型调用 0、下载 0。CLI `--help` 成功，退出码 0。
- `test_server_cli_contract` 剩余 5 项通过（0.041 秒）；未跑全套历史测试。TypeScript `--noEmit --incremental false` 通过。
- 使用现有 Node 22 执行一次 `run-framework.mjs build` 成功；新生成的有效 `web/dist/` 保留。
- 本地 Vite 页面在 5199 返回首页 HTTP 200、HTML 30,977 B，含职业文本；未启动后端、模型调用 0。主 Agent 已 SIGTERM 并等待其前端进程组退出。
- 未进行真实 GPU/DeepSeek 调用、训练、下载、Sites 部署或 Git 发布。

### 临时文件与空间记录

本 Agent 仅清理本次允许的 `web/node_modules/.vite/`、`web/node_modules/.vite-temp/`、`web/.wrangler/`、`web/.vinext/`、`web/tsconfig.tsbuildinfo`（存在时）及 `/tmp/career-cleanup-execution-20261110/`。保留新 `web/dist/`、`web/.next/types/routes.d.ts`、Sites runtime profile、`.openai` 配置与有效依赖。清理前基线为 `du -sx -B1 .` = 7,262,867,456 B。待完成上述清理后仅测一次同一口径；本记录回写可能令占用少量变化，不再复测。

- 清理后实测 `du -sx -B1 .`：6,748,827,648 B；较清理前 7,262,867,456 B 基线减少 514,039,808 B（约 490.2 MiB）。测量后仅写入本行记录，按要求未复测；最终占用可能有少量差异。

## 当前安排

### 编号 1 执行记录（2026-11-10）

| 时间 / 工作 | Agent / 模型 | 文件与结果 |
|---|---|---|
| 2026-11-10 16:32（Asia/Shanghai），启动第 1 项功能收尾 | 主 Agent `/root` 调度 `/root/stage1_frontend_gaps`、`/root/stage1_backend_gaps`（均 `gpt-6-luna max`）做只读定向功能链扫描；记录代理 `/root/stage1_record_start`，`gpt-6-luna max` | 复用此前已完成的文件精简成果；计划前端、后端沿实际演示主链路扫描，结果待回报，发现实际复杂问题时再派 `gpt-6-sol xhigh`。本记录代理仅修改 `DEMO_PLAN.md`、`AGENT_WORKLOG.md` 的状态和启动记录。未重扫全仓库、未再次清理、未跑全套测试；GPU/DeepSeek 调用和部署均未执行。实际检查：只读 `AGENTS.md`、`DEMO_PLAN.md`、`AGENT_WORKLOG.md`；未运行测试或构建。问题/依赖：无新增阻塞；后续按定向扫描实际结果处理。 |
| 2026-11-10 16:33（Asia/Shanghai），更新合同中的过时分工与模型调用预算 | `/root/stage1_record_start`，`gpt-6-luna max` | 修改 `WEB_CONTRACT.md`：将旧 Agent A/B/C 分工更新为当前主 Agent、Luna max 与 Sol xhigh 职责，增加计划/工作记录链接；将旧真实验收调用数量改为仅第 3 项按 `DEMO_PLAN` 当前预算执行、第 1/2 项不调用。更新本表记录本次修改。未运行测试或构建，未调用项目模型。 |
| 2026-11-10 16:37（Asia/Shanghai），完成第 1 项功能收尾与结果记录 | `/root/stage1_frontend_gaps`、`/root/stage1_backend_gaps` 均为 `gpt-6-luna max`；文档代理 `/root/stage1_record_start`，`gpt-6-luna max` | 两路只读定向核对将已有入口/接口对照到材料入口与上传 → 追问 → 规划 → 反馈 → 图谱/来源 → 导出 → 恢复，未发现确定的主链路阻断缺陷；不加功能、不改业务代码、不启动 Sol。`dataset/demo/anonymous-export.json` 是匿名静态导出参考，不是导入数据或已恢复的在线会话；“我的记录”恢复 `data/local-demo/` 中已保存的会话。此前文件精简为前序成果，不计作本轮新成果。记录代理本轮更新 `DEMO_PLAN.md`、`AGENT_WORKLOG.md`、`README.md`；本阶段合并记录还包括先前对 `WEB_CONTRACT.md` 的预算引用及主 Agent/Luna/Sol 分工对齐。只读定向代码/接口对照；此前 `git diff --check` 已通过。未运行测试、构建或浏览器试跑，未调用 GPU/DeepSeek，未训练、下载、部署或发布 Git；未重跑历史清理检查，无运行中间文件。问题/依赖：Sites 原生工具和现有 HTTPS 入口仍缺；第 2 项待执行。 |

第 1 项功能收尾已完成：两路静态定向核对未发现确定的主链路阻断缺陷，未修改业务代码；没有进行真实运行或模型验证。上文编号 2 的 Sites 准备和编号 3 的 Demo 试跑是此前计划状态；当前工作按用户最新指示转为 GitHub 源码交付与本机 Sites 交接，Demo 验证已确认完成，不再重复。

### 编号 2 下载准备记录（2026-11-10）

| 时间 / 工作 | Agent / 模型 | 文件与结果 |
|---|---|---|
| 2026-11-10，Cloudflare 官方独立程序下载 | `/root/download_cloudflared_task2`，`gpt-6-luna max` | 用户明确授权本次下载。Linux x86_64 对应官方 `linux-amd64`，通过 Cloudflare 官方 latest 下载链接取得 `data/runtime/cloudflared-task2-20261110T101433Z/bin/cloudflared`，40,274,445 字节、程序权限 700；专用任务目录及子目录权限 700，`config/download.json` 与日志权限 600。首次受限网络请求以 curl 7 / HTTP 000 失败，残缺 tmp 已删除；自动审批通过后使用同一官方 latest 链接，HTTP 200 下载成功。来源、平台、架构、完成时间和文件大小记录在专用目录的 `config/download.json`，日志也保存在该目录。未查询版本、未做 SHA 校验、未执行程序、未启动隧道、未调用模型或部署；现有 API/Sites 配置和服务令牌未改。该文件保留至后续第 3 项功能测试结束，再统一删除。后续仅待授权按临时隧道方案完善启动/清理并启动；本次只完成下载准备，不记为 Sites 完成。当前主仓库变更未提交或推送。 |

第 2 项的 cloudflared 下载准备已完成，但 Sites 部署仍待执行；依赖 Sites 原生能力，API/临时隧道尚未启动。后续等待授权完善临时隧道的启动与清理后再启动，再进入第 3 项桌面试跑。第 3 项预算为本地引导最多 2 次、DeepSeek 规划 HTTP 最多 1 次、本地反馈最多 1 次、视觉 0 次、自动重试 0 次；仅进入第 3 项后启用。

以上是此前编号计划的历史状态。依据用户最新授权，当前不启动 API 或临时隧道；由本机 Sites 工作流在后续完成部署。

### GitHub → 本机 Sites 源码交接准备（2026-11-10）

| 时间 / 工作 | Agent / 模型 | 文件与结果 |
|---|---|---|
| 只读源码边界与构建可移植性扫描 | `/root/github_source_inventory`、`/root/sites_build_portability_read`，均为 `gpt-6-luna max` | 确认现有 `web/` 的 React、Vite/Vinext 与 Cloudflare Worker 架构符合当前 Sites Worker 部署形态，Worker 构建输出为 `dist/server/index.js` 默认 fetch handler，保留该入口才能继续使用 `/api` 代理。干净克隆缺少 `.sites-runtime` 时 `readExecutionProfile` 回退到 portable profile；构建无需服务器 Python/CUDA/权重或本地 profile，也不依赖 `bash`/`timeout`。该扫描只读，未构建、测试、启动 API、调用模型或部署；没有声称 Windows/macOS 安装已实测成功。 |
| 干净 web 源码构建 | 主 Agent `/root` | 对已暂存的 82 个 web 源文件制作干净副本，不带 `.env`、`.sites-runtime`、`next-env`、`.next` 或 `dist`；复用服务器已有 `node_modules`，没有重新安装或下载依赖。在 `/tmp/career-sites-handoff-h43j3q1g/web` 使用 Node 22 执行 `npm run build`，退出码 0；`/` 与 `/api/:path+` 路由编译成功，生成 `dist/server/index.js`（198,266 B）、`dist/client` 和 `dist/.openai/hosting.json`。这是 Linux 本服务器上的干净源码构建验证，不是新装依赖验证或 Windows/macOS 实测；0 模型调用、0 浏览器试跑、未部署。临时构建目录留给主 Agent 后续清理。 |
| 文档与忽略规则更新 | `/root/write_github_sites_handoff`，`gpt-6-luna max` | 修改根 `.gitignore`，补充 SQLite 旁路文件和常见私钥扩展名规则；更新 `AGENTS.md`、`README.md`，新增 `SITES_HANDOFF.md`，记录 GitHub 克隆/构建、本机 Sites 项目目录 `web/`、Worker 输出、服务端运行变量及部署边界。当前 `web/.openai/hosting.json` 尚未登记 `project_id`，`d1` 和 `r2` 保持 `null`；桌面注册后使用实际 ID。文档代理未运行测试、构建、安装或生成额外文件；未改业务源码、真实 `.env`、用户数据或 `web/.gitignore`。提交/推送等待主 Agent 实际执行和记录。 |
