# 成员B 工作项5/7 验收记录：三档输出调整 + 实验4（动态输出）

更新时间：2026-09-08 ｜ 分支：`work/member-b-feedback-exp4` ｜ Prompt 版本：planning-v1 / feedback-v1

## 做了什么

### 工作项5：过简/适中/过详三档输出调整（原手册第5周末节点）

- 新增 `project/core/feedback_prompt.py`（版本 feedback-v1）：三档反馈中，
  「过短」指令扩充（差距≥4条、每阶段交付物≥3条、行动≥5条、建议≥原1.5倍），
  「过于详细」指令精简（建议≤200字、差距≤3条、交付物≤2条、行动≤3条），
  「合适」不触发重新生成；两档都保持与 planning-v1 相同的严格 JSON 契约、
  30/90/180 三阶段结构和"引用知识库事实、不虚构"要求。
- 复用 planning_prompt.py 抽出的共享 schema/要求常量，避免两套 Prompt 漂移
  （渲染结果逐字不变，planning-v1 版本不变，原 9 个 Prompt 测试仍全过）。
- 新增 `CareerOrchestrator.adjust_plan(session_id, feedback)`：
  用原规划 JSON + 保存的管线上下文重建 Prompt，沿用与主流程相同的重试策略
  （网络类错误重试、JSON/路线校验）；失败时按手册 3.2 **保留原结果并记录失败**，
  错误写入 run 日志（`run_logging` 新增可选 `error`/`feedback_adjusted` 字段，
  向后兼容）。
- CLI（assistant_cli）反馈交互接通：过短/过于详细会真正重新生成并提示，
  合适仍走原记录路径；API 的 `submit_feedback` 接口行为不变。
- `DeepSeekBrainClient` 记录最近一次调用的 token 用量（`last_usage`），
  满足手册 6.1 对 Token 用量的记录要求。

### 工作项7：实验4 动态输出（原手册第7周节点）

- 新增运行器 `project/experiments/run_dynamic_output_experiments.py` 和
  20 个固定匿名样例 `project/experiments/dynamic_output_cases.json`
  （10×过短 + 10×过于详细；「合适」不触发调整，等价于统一组，故不单独设组）。
- 实验设计：配对设计，每例跑两组——统一输出（A组）与按反馈调整（B组），
  20 例 × 2 组 = 40 行；单一变量=反馈调整，Prompt/模型/检索配置冻结。
- 自动记录（手册 6.1）：输入、输出全文、耗时、错误、检索命中、Token 用量，
  以及档位细节指标（建议字数、差距数、交付物数、行动数、内容量）。
- 客观指标：**方向匹配（接收效率）**——过短档调整后内容量应增大、过于详细档
  应减小；**档位差异可辨**——两组均值差异明显即可视化。
- 离线模式使用确定性的假客户端（档位长度可辨），仅验证流程，行内标记
  `offline-fake`，**不计入实验数据**（手册数据纪律）。
- 人工评分：`scores_template.csv` 供成员C按 5 分制盲评
  （信息完整性/建议相关性/实用性/输出适配度=适配度与反馈评分），评分时隐藏组别。

## 真实实验结果（2026-09-08，deepseek-v4-flash，40 次调用）

| 档位 | 统一输出均值(字) | 调整后均值(字) | 方向匹配率 | 失败数 |
|---|---|---|---|---|
| 过短 | 1174.2 | 2337.8（约 2.0 倍） | 10/10 | 0 |
| 过于详细 | 961.3 | 389.4（约 0.41 倍） | 10/10 | 0 |

- 40/40 行 `served_by=cloud_brain`（真实模型，无模板降级），`adjust_error` 全空。
- 全部 20 例知识库检索命中（knowledge_hits>0）；总 token 用量 99,420；
  平均单次调用耗时约 10.8s。
- 版本化材料在 `docs/exp4-b/`：results.json/CSV（原始数据）、summary.json/md、
  chart.svg（图表）、scores_template.csv（盲评表）。
- 运行日志 `data/experiments/dynamic_output-live/experiment_runs.jsonl`
  按仓库约定留在忽略目录，可在新环境用同一命令复现。

## 如何验证

```powershell
python -m unittest discover -s project/tests -v
python -m project.experiments.run_dynamic_output_experiments --output-dir data/experiments/dynamic_output-offline-check
python -m project.experiments.run_dynamic_output_experiments --live --output-dir data/experiments/dynamic_output-live
```

新增 28 个测试全过（feedback-v1 Prompt 契约、adjust_plan 成功/失败/合适/无待调整、
Token 用量记录、实验样例文件与离线运行器产物）。离线检查输出 40 行且全部
标记 `offline-fake`。

## 已知事项

- 全量 unittest 中 8 个既有用例在本机 Windows 上报 sqlite 临时文件占用错误，
  已在干净基线 `integration/week1-results`（worktree）复现确认与本分支改动无关，
  属 `SessionMemory` 连接未显式关闭的既有问题（建议后续由成员A/负责人统一修）。
- 人工评分完成后，B 需按评分表汇总均值/差异并更新本文件结论。
- 档位指令中的篇幅阈值（1.5 倍、200 字等）属反馈-v1 冻结口径，如实验显示
  不合适，需升版本 feedback-v2 并重跑实验。
