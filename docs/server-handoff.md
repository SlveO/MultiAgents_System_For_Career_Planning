# 新对话启动指令

主责：服务器 Agent。唯一仓库：`/home/shenxiaoyang/project/大创/MultiAgents_System_For_Career_Planning`；父目录不是 Git 仓库。先检查 Git 状态、main 与远端引用；保留已有任务分支与未提交修改。仅在工作树干净且开始新任务时从 main 建 `codex/` 短期分支。

唯一目标是原版职业规划 CLI（`project/assistant_cli.py`）。最终推理使用 DeepSeek，其余模型功能使用本地 GPU。禁止恢复旧对照实验、论文路线、Web/API 产品栈。需求引导开发基线为 `4a76edaa3ec333bb4e709673c9478cd8c764f333`；本版发布的最终 main 完整哈希见交付消息及服务器 `data/maintenance/local-needs-guidance-publication-20261002/publication.json`。此前归一与清理提交仅作来源记录。

已验证同次图片→L20 cuda:1→DeepSeek→CLI 展示与 SQLite/JSONL 持久化成功，15.54 秒，视觉/API 各调用一次；61 项离线测试通过。原验收脚本误判记录保留，离线复核确认融合事实进入实际输入，并非重新运行。人工质量尚未评分。需要证据时阅读[验证摘要](cli-e2e-verification-20261001.md)，不要重复冒烟。

本版包含本地需求引导、回答状态/来源合同和 CLI 接入。2026-10-02 已按授权完成三例真实 Qwen3-4B 引导验证及修正，85 项离线测试通过；见[GPU验收](local-needs-guidance-gpu-20261002.md)。负责人已授权提交并发布；发布是否成功以完整哈希记录为准，不凭本文推断。下一任务为本地反馈输出适配：先检查三档反馈、规划合同与现有存储，设计仅改变表达详略的最小方案，再做实现和离线回归。不要重复此次 GPU 验证或自动调用 API。`GUIDANCE_MODEL_PATH` 仅在验证进程中配置，未写入私有配置。需求引导的小样本验收不代表长期稳定性或人工质量评分。任务表见[完成计划](completion-plan.md)，期限待负责人提供。

复用 `.venv-l20/bin/python`，设备 L20 cuda:1；私有配置仅在服务器读取，不打印密钥、不安装依赖、不下载权重、不恢复 GPU 使用权门禁。真实 GPU 作业前仍须按仓库上级规范检查实时设备占用；本交接不构成无限 GPU/API 调用授权。

清理已经完成，不重复清理。模型、环境、`.env`、数据库、真实运行证据和原 41 项修改 stash 均保留。历史资料集中于忽略目录 `data/maintenance/history-before-20261001/`，按需查阅，不全量读入上下文。清单与恢复方式见[清理回执](workspace-cleanup-20261001.md)。后续发布不强推、不绕过保护规则；接收时回报最终 main 完整哈希与首项任务。

成员 C 的固定资料提交已保存并审阅，未合入或采用；需要参考时先读[接收回执](member-c-receipt-20261001.md)，不要直接导入历史规则或岗位排序。服务器认证维护见[持久化认证说明](github-authentication.md)。
