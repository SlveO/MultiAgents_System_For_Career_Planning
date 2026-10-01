# 新对话启动指令

主责：服务器 Agent。唯一仓库：`/home/shenxiaoyang/project/大创/MultiAgents_System_For_Career_Planning`；父目录不是 Git 仓库。先检查 Git 状态、main 与远端提交，从 main 建 `codex/` 短期分支；不要覆盖现有修改。

唯一目标是原版职业规划 CLI（`project/assistant_cli.py`）。最终推理使用 DeepSeek，其余模型功能使用本地 GPU。禁止恢复旧对照实验、论文路线、Web/API 产品栈。产品源码基线为 `6190f0f53ea8e042760a8867383e0710ff354680`；包含本交接文档的最终 main 完整哈希见交付消息及服务器 `data/maintenance/workspace-cleanup-20261001/publication.json`。

已验证同次图片→L20 cuda:1→DeepSeek→CLI 展示与 SQLite/JSONL 持久化成功，15.54 秒，视觉/API 各调用一次；61 项离线测试通过。原验收脚本误判记录保留，离线复核确认融合事实进入实际输入，并非重新运行。人工质量尚未评分。需要证据时阅读[验证摘要](cli-e2e-verification-20261001.md)，不要重复冒烟。

下一任务是本地需求引导：先检查固定八题入口、`project/core/schemas.py` 画像合同和已有本地模型，提出最小实现方案。交付物为引导模块及离线回归测试；依赖已有模型确认与执行范围，期限待负责人提供。验收：区分未知与缺失；针对缺失或矛盾提问；不重复问已回答内容；回答可追溯进入规划。完成后才做反馈输出适配。任务表见[完成计划](completion-plan.md)。本次交接未开发新模块。

复用 `.venv-l20/bin/python`，设备 L20 cuda:1；私有配置仅在服务器读取，不打印密钥、不安装依赖、不下载权重、不恢复 GPU 使用权门禁。真实 GPU 作业前仍须按仓库上级规范检查实时设备占用；本交接不构成无限 GPU/API 调用授权。

清理已经完成，不重复清理。模型、环境、`.env`、数据库、真实运行证据和原 41 项修改 stash 均保留。历史资料集中于忽略目录 `data/maintenance/history-before-20261001/`，按需查阅，不全量读入上下文。清单与恢复方式见[清理回执](workspace-cleanup-20261001.md)。后续发布不强推、不绕过保护规则；接收时回报最终 main 完整哈希与首项任务。

成员 C 的固定资料提交已保存并审阅，未合入或采用；需要参考时先读[接收回执](member-c-receipt-20261001.md)，不要直接导入历史规则或岗位排序。服务器认证维护见[持久化认证说明](github-authentication.md)。
