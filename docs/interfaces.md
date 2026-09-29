# 当前接口

权威Python合同：project/core/schemas.py；执行边界：dataset/completion_protocol.json。

- TaskRequest：会话ID、用户目标、输入材料、约束、追问回答。CLI只走DeepSeek规划。
- PerceptionResult：模态、摘要、事实、证据、置信度、缺失信息。不能把置信度当效果评分。
- UserProfile：学历、专业、技能、兴趣、岗位目标、预算、偏好、限制。
- CareerPlanResponse：目标岗位、差距、30/90/180天路线、资源、行动、风险和建议。
- submit_feedback：当前只记录过短/合适/过于详细，不生成新规划。输出适配需后续新增明确接口与回归测试。

图片代理调用GPU本地ImageProcessor；CUDA/权重校验失败不触发CPU加载。
DeepSeek失败可在内部产生明确标注的诊断模板，但CLI拒绝将其报告为成功规划，退出码为1。
请求与响应入库时删除原文、证据路径和引文等字段；真实隐私边界仍需案例验证。

后续本地引导和适配智能体应复用这些字段，不恢复旧research-*架构比较Schema。
