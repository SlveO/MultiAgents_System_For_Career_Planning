# 服务器代码归一记录

日期：2026-10-01。负责人已授权测试、提交、正常合入 main 和推送；只有 main 推送核验后才清理已确认废弃分支与工作区。禁止强推、reset --hard、git clean 或覆盖未知修改。本机文件不在操作范围。

## 有效成果与来源

接收基线 a092c045dd02c835e991796ab3140d034c2e7ca7；有效增量是 DeepSeek 代理选择修复、六项客户端回归测试、真实验证摘要与接管文档。提示词、推理参数及产品功能不扩展。

原开发目录的 41 项状态（17 修改、23 未跟踪、1 删除）已与 7995c628595300bd67b122b19df7571ec57e138b 来源快照核对；不将旧快照整体合回 main。既有来源清单和恢复资料保留在服务器，不重复打包。

初始 main 为 c30a0e71222f674f65354e4c5592d28ce04e4e58，是已接收基线的祖先；本轮 fetch 后未发现 main 独有提交。GitHub 默认分支为 main。最终发布完整哈希和清理回执以推送后核验记录为准。

### 历史检查点：认证阻塞（现已解除，最终状态见下文）

有效成果已提交为 a3e490fa5c48cfd4ae4fbfaa743af6e39b672b84，本地 main 已正常快进到该提交。本记录的后续文档提交同样尚待发布。执行 `git push origin main` 时返回 `could not read Username for 'https://github.com': No such device or address`（环境显示本地化错误）；未能建立写入认证，不能据此判断分支保护是否允许直接推送。

未配置可用的 credential helper，未发现 gh 安装或本地 SSH 密钥目录。需负责人在服务器配置 GitHub 认证，不要将 token 发到对话或提交 Git。认证就绪后先重新 fetch，复核远端差异，再正常推送；如保护规则要求 PR 则遵循 PR 流程。

尚未删除任何远端分支、工作区或文件。主仓库原 41 项修改、环境和数据原样保留；接收工作区的 .env、数据库和四组真实验证记录亦原位保留。迁移和清理须等待 main 推送核验成功。本机暂不可把未发布的新 main 当作恢复来源，也不应据此删除本机仓库。

## 本轮离线测试

使用既有 .venv-l20/bin/python（Python 3.10 环境），不安装/更新依赖：

- `python -m project.assistant_cli --help`：通过。
- `python -m compileall -q project scripts/models`：通过。
- `python -m unittest discover -s project/tests -v`：61 项通过，测试计时 0.666 秒。
- `git diff --check`：通过。

原日志保存在服务器 `data/maintenance/consolidation-20261001/`。本次 GPU/API 调用均为零；没有以删除测试制造通过。

## 真实验证范围

原始证据在 `data/acceptance/` 的 cli-once-20260930、cli-proxy-fix-20260930、network-check-20260930、deepseek-only-20260930 四目录。前两项对应 a092c04 及其客户端修复工作树；后两项诊断/请求使用同一修复代码，详见 deepseek-client-verification-20260930.md。

图片在本地 L20 cuda:1 提取事实成功；同次 CLI 未生成规划。单独 DeepSeek 请求复用保存的感知输入，6.67 秒返回可解析规划，路线结构校验通过。没有同次完整 CLI 成功证据，没有人工质量评分，不能把结构通过解释为建议正确。

## 资源与清理边界

保留主仓库 MultiAgents_System_For_Career_Planning 的共用 .git、.venv-l20、历史 data；模型目录不移动、不上传。当前接收工作区的 .env、真实结果与数据库须迁移并逐文件校验后才能移除；同名不同内容不得覆盖。数据库碰撞须独立保存。

旧发布工作区已检查为干净且无未跟踪/忽略文件；失效 /tmp worktree 仅清理注册信息。父目录既有来源归档、接收回执及交接文档保留，不把未知文件当垃圾删除。

拟保留 main；既有 12 个非 main 远端分支属于已放弃研究、成员、审阅或交付分支，须在发布核验后逐项删除。若出现未知新分支或独有修改则保留并报告。保护规则不关闭、不绕过。

## 下一项与本机回执

服务器 Agent 先完成同次图片→GPU→DeepSeek→CLI展示与持久化验证，验收输入链可追溯、真实规划校验通过、失败不伪装成功。依赖 main 归一、私有配置、设备可用及真实运行范围确认；本次不启动。之后本地需求引导，再反馈输出适配。

GitHub main 的最终核验完整 commit 是代码恢复来源，但不包含权重、环境、密钥和运行数据。本机 Agent 必须先检查独有 log.txt、output/、data/、models/、配置与未提交内容，确认必要文件已保留后才可删除本机仓库。原始申报书和结题材料不在删除范围内。

## 最终清理结果

负责人配置仓库限定的认证后，main 正常推送到 cf0dabb1c1f5e4d4136a28041b24b4bb96040337，远端完整哈希和默认分支 main 已核对。本节文档随后作为最终回执提交并正常推送；最终完整 commit 以交付消息及远端 main 为准。没有强推，没有修改或绕过分支保护。

远端仅保留 main，已核对并删除以下 12 个已放弃分支（删除前提交无漂移）：

- codex/original-proposal-cleanup-20260929
- codex/server-inventory-20260927
- completion-mvp
- feature/b-week1
- integration/week1-results
- review/phase-c-20260825
- web
- work/lead-evaluation
- work/member-a-experiment-runner
- work/member-b-feedback-exp4
- work/member-b-l20-models
- work/member-b-planning-prompt

唯一开发工作区为 MultiAgents_System_For_Career_Planning，使用既有 .venv-l20/bin/python（Python 3.10.12）。original-proposal-server-20260929 与 server-inventory-publish-20260928 均经检查后用 git worktree remove 正常移除；/tmp/phase-c-github-review-20260825 原目录不存在，仅 prune 失效注册信息。未使用 force、reset 或 clean。

原 41 项修改已保存到 stash 23e04aeceb4ee187a0067e06ee217a765b4dab76，再正常切换 main；它们与来源快照一致，不作为待合入的新功能。历史本地分支保留为来源恢复引用，不是并行开发端。父目录既有来源归档、接收回执、交接资料及工具配置保留，未新建重复归档或删除未知文件。

迁移的 39 个文件逐文件 SHA-256 校验通过，记录在 data/maintenance/consolidation-20261001/migration.json。包括 .env（权限 0600）、四组 data/acceptance/ 真实验证资料、测试日志、测试临时资料与两个数据库。原主目录数据库改存 data/acceptance/legacy-session-before-takeover.db；接收目录数据库迁为 data/session_memory.db，未覆盖任何旧数据库。历史运行脚本内旧路径保留作证据，不保证可在新目录直接重放。

模型、.venv-l20、历史 data 及个人配置未上传 Git。工作树无待提交源码/未跟踪源码，忽略项为配置、环境、数据、测试临时目录与 Python 缓存。迁移后 help、compileall、61 项测试再次通过（0.616 秒），原日志 final-*.log 位于同一 maintenance 目录。GPU/API 本次调用为零。

服务器侧清理已完成。当前无清理权限阻塞或待确认的新远端分支；既有非工作区资料有意保留。下一项仍是同次完整多模态 CLI 验证，不将独立 API 成功当作已完成；本轮不开发需求引导或反馈适配。本机文件未操作，本机删除前仍须完成前述独有文件核对。
