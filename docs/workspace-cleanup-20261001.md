# 工作区清理回执

2026-10-01，服务器 Agent 按已授权计划完成文件整理与交接；产品源码仍为 `6190f0f53ea8e042760a8867383e0710ff354680`，未修改推理功能、未启动需求引导开发，GPU/API 调用为零。本机文件未操作。

## 清理与恢复

父目录只保留实际仓库、`START-HERE.md` 入口及原受保护隐藏目录。以下六项迁入仓库忽略目录 `data/maintenance/history-before-20261001/`，共 35 个文件，逐文件 SHA-256、大小与权限核对一致，无覆盖：

- `server-inventory-20260927/`
- `server-receipt-20260929/`
- `server-review-20260927.zip`
- `server-review-20260927.zip.sha256`
- `member-b-l20-agent-handoff.md`
- `server-transition-execution-guide-20260927.zh-CN.md`

重复目录 `server-review-20260927/` 删除前再次核验：188 个交付文件与 ZIP 内容逐字节一致，其余 66 个文件均为 Python 缓存。ZIP 校验值与 CRC 检查通过；需要恢复时将保留 ZIP 解压至新的空目录，不覆盖当前仓库。

源码区清除 139 个 Python 缓存及 18 个测试临时文件。测试数据库位于 `project/data/test_tmp/memory_*/mem.db`，只含测试 `s1` 的固定画像、反馈和交互；知识文件位于同级 `knowledge_*/kb.json`，来源为现有组件测试。清空后移除旧 `project/api/`、`project/experiments/`、`scripts/research/` 及临时空目录；离线检查新生成的缓存和测试文件也已清除。未遍历清理虚拟环境或历史证据包。

模型目录及权重文件仍存在（未重新做全量权重校验）；`.venv-l20`、`.env`、实际数据库、`data/acceptance/`、`data/experiments/`、`data/handoff/` 保留，受保护数据逐文件哈希复核无变化。原 41 项修改仍在 stash `23e04aeceb4ee187a0067e06ee217a765b4dab76`，不能整体恢复旧研究功能。

## 验证与发布记录

使用既有 `.venv-l20/bin/python` 执行 README 的 CLI help、compileall 与 unittest：全部通过，61 项测试，0.498 秒。差异检查在提交前执行。最新真实链路结果见[验证报告](cli-e2e-verification-20261001.md)，不将离线测试描述为新的 GPU/API 验证。

服务器审计目录为 `data/maintenance/workspace-cleanup-20261001/`：`migration.json` 记录来源、目标、权限和哈希；`deletion.json` 与 `post-check-deletion.json` 记录删除明细；`protected-before.json`、`final-verification.json` 记录保留核验；三个 `.log` 文件记录离线检查。最终发布完整哈希和远端一致性写入推送后的 `publication.json` 并在交付消息报告。若远端出现新修改、冲突、认证或保护阻塞，停止发布，不强推。

新对话按[启动指令](server-handoff.md)接收最终 main 哈希，先检查固定八题、画像合同及已有模型，提出本地需求引导最小方案；不重复清理与冒烟。
