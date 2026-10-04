# 新对话启动指令

主责：服务器 Agent。唯一仓库：`/home/shenxiaoyang/project/大创/MultiAgents_System_For_Career_Planning`；父目录不是 Git 仓库。先检查 Git 状态、main 与远端引用；保留已有任务分支与未提交修改。仅在工作树干净且开始新任务时从 main 建 `codex/` 短期分支。

唯一目标是原版职业规划 CLI（`project/assistant_cli.py`）。最终推理使用 DeepSeek，其余模型功能使用本地 GPU。禁止恢复旧对照实验、论文路线、Web/API 产品栈。需求引导开发基线为 `4a76edaa3ec333bb4e709673c9478cd8c764f333`；当前已发布 main 为 `936929a4c42c0fb5406ab76531102503e1bbfab0`，远端已核验，发布回执在 `data/maintenance/local-feedback-publication-20261002/publication.json`。此前归一与清理提交仅作来源记录。

已验证同次图片→L20 cuda:1→DeepSeek→CLI 展示与 SQLite/JSONL 持久化成功，15.54 秒，视觉/API 各调用一次；61 项离线测试通过。原验收脚本误判记录保留，离线复核确认融合事实进入实际输入，并非重新运行。人工质量尚未评分。需要证据时阅读[验证摘要](cli-e2e-verification-20261001.md)，不要重复冒烟。

本版包含本地需求引导、回答状态/来源合同和 CLI 接入。2026-10-02 已按授权完成三例真实 Qwen3-4B 引导验证及修正，85 项离线测试通过；见[GPU验收](local-needs-guidance-gpu-20261002.md)。需求引导已正常发布，main 完整哈希核验为 `3308d65c3941e31c0c60a174e9d4fc37f22c8989`。

反馈模块、100 项离线回归及三档 GPU 小样本已完成并按授权发布到上述 main。首轮仅扩充通过，压缩前利用率采样触发停止；原脚本及失败保留，占用检查修正并通过六项模拟检查。负责人另行授权仅补测压缩/合适，最终扩充12→247字、压缩409→207字、合适原文零模型，原事实及 CLI/SQLite/JSONL 版本一致。两轮合计两次已有 Qwen3-4B 生成，最终规划模拟，零 API；发布21个文件与审阅哈希一致，不重复测试或 GPU 验证。详见[实施记录](local-feedback-adaptation-implementation.md)与[GPU 三档验收](local-feedback-gpu-20261002.md)。

当前短期分支 `codex/final-cli-demo` 从发布后的干净 main 建立。最终演示已于北京时间 2026-10-05 按负责人另行授权执行 attempt-2：约 45.37 秒、退出码 0、18 项检查通过，一次真实 DeepSeek API 规划接一次 L20 cuda:1/Qwen3-4B反馈生成。CLI 752→2204 字，原规划不变，两版关联及 SQLite/JSONL 一致。产品源码未变，复用 100 项离线回归。首轮预检失败 API/模型均 0，原记录保留；CUDA 初始化顺序修正已通过，首轮具体异常消息仍不可追溯确认。见[真实验收](final-cli-demo-verification-20261005.md)和[方案与修正历史](final-cli-demo-plan-20261002.md)，不要重复运行。

本分支验收文档、报告草稿及 `dataset/completion_protocol.json` 完成状态已更新，尚未提交或发布；保留修改，不切回 main 覆盖。主责：服务器 Agent；交付：可追溯验收摘要与报告更新；依赖：负责人在 diff/检查审阅后单独授权提交、快进 main 和正常推送，期限待提供；验收：仅审阅文件入提交、源码不变、失败记录保留、远端完整哈希一致；下一步：获 Git 发布授权后发布并记录回执。GPU/API 运行授权不包含发布，当前已发布 main 仍为上述 `936929a`。

结题报告与剩余要求见[报告](completion-report-draft.zh-CN.md)；负责人已回复学校截止日期、模板和必交成果“待确定”，人工评价未采集。不要重复已完成的 GPU 验证或自动调用 API。`GUIDANCE_MODEL_PATH` 和 `FEEDBACK_MODEL_PATH` 仅在验证进程中配置，未写入私有配置。小样本验收不代表长期稳定性或人工质量评分。任务表见[完成计划](completion-plan.md)，期限待负责人提供。

复用 `.venv-l20/bin/python`，设备 L20 cuda:1；私有配置仅在服务器读取，不打印密钥、不安装依赖、不下载权重、不恢复 GPU 使用权门禁。真实 GPU 作业前仍须按仓库上级规范检查实时设备占用；本交接不构成无限 GPU/API 调用授权。

清理已经完成，不重复清理。模型、环境、`.env`、数据库、真实运行证据和原 41 项修改 stash 均保留。历史资料集中于忽略目录 `data/maintenance/history-before-20261001/`，按需查阅，不全量读入上下文。清单与恢复方式见[清理回执](workspace-cleanup-20261001.md)。后续发布不强推、不绕过保护规则；接收时回报最终 main 完整哈希与首项任务。

成员 C 的固定资料提交已保存并审阅，未合入或采用；需要参考时先读[接收回执](member-c-receipt-20261001.md)，不要直接导入历史规则或岗位排序。服务器认证维护见[持久化认证说明](github-authentication.md)。
