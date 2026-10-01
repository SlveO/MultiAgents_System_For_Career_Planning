# 单次完整 CLI 验证

日期：2026-10-01。执行代码为 main 基线 6190f0f53ea8e042760a8867383e0710ff354680，分支 codex/cli-e2e-verification-20261001。产品代码、提示词、模型配置均未修改。负责人授权本次下一任务；没有自动追加模型调用。

## 结果

同一次图片→本地 Qwen3-VL-2B（L20 cuda:1）→DeepSeek→CLI 展示及持久化通过。真实 CLI 返回 0，总耗时约 15.54 秒；视觉约 9.77 秒。视觉调用一次、API 调用一次，零重试，禁止隐式下载和 CPU 模型回退。

输入是既有匿名 image-001 与合成统计学大三学生画像。模型输出真实规划，既有路线结构校验及 CareerPlanResponse 校验通过；CLI 展示建议与 API 解析内容一致，SQLite 响应等于脱敏后的 CLI 输出，JSONL 建议和服务来源一致。独立 SQLite 与 JSONL 保存在本次目录，不污染默认会话数据库。反馈使用合成选项“2”仅验证记录流程，rating 为空，不计满意度或人工评分。

## 原始验收脚本问题与离线复核

原 result.json 保留 pipeline_success=false，原因只有一项：验收脚本错误要求整段视觉原文直接包含在 API 输入里。真实管线按既有 ImagePerceptionAgent 提取 facts，再通过 MultiModalFusion 融合，设计上不直接注入整段 raw_output。不能把该脚本判定描述为产品执行失败，也不能覆盖原判定伪装一次全绿。

新增 verify_saved_chain.py 仅读取已有产物并复用既有解析/融合函数，不调用模型：原产物哈希全部通过；视觉原文与响应 raw_output 相等；提取事实与响应 facts 相等；重建融合文本确实出现在实际 API prompt；prompt SHA-256 一致。其他六项原检查全部通过，因此 offline-chain-verification.json 给出 pipeline_success=true。未修改原 execute_once.py、result.json 或模型原文，未重跑 GPU/API。

## 证据与局限

服务器目录：data/acceptance/cli-e2e-20261001T054808Z/。

- command.json：代码版本、核心源码哈希、图片哈希、实际 CLI 参数及解释器。
- vision.json、brain-input.json、brain-output.json：原始链、设备、耗时、输入/输出哈希。
- cli-output.json、cli-response.json、parsed-plan.json：CLI 展示及规划。
- session.db、runs.jsonl：本次真实持久化和合成反馈。
- result.json、manifest.json、offline-chain-verification.json：原判定、哈希清单和离线复核。
- execution.log、execute_once.py、verify_saved_chain.py：执行记录和审计脚本。

图片内容包含 SQL、仪表板、周指标报告、在读本科生四项实质信息。规则解析还保留一条开场句和部分 Markdown 标记；不把五条解析文本称作五条独立事实。规划将“未提供 SQL 水平”断言为“缺少 SQL 能力”，并使用受外部因素影响的 offer/面试指标，说明语义质量仍有局限。未开展人工质量评分、音频/文档/流式验证或反馈适配验证。历史运行资料不上传 Git。

## 下一项

主责：服务器 Agent。交付物：本地 GPU 需求引导模块及离线回归测试。依赖：本次完整链路结果接收、现有本地模型选型与执行范围确认。验收：区分未知与缺失，缺失/矛盾信息触发针对性提问，已有回答不重复问，回答可追溯进入最终规划。随后才实现反馈后的输出适配。本轮只验证现有链路，未开发这两个模块。
