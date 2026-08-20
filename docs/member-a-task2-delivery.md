# 成员 A 第二次任务交付说明

- **分支**：`work/member-a-experiment-runner`（基于 `origin/integration/week1-results`）
- **提交**：`fecf84e`

## 交付内容

3 个新增文件，未修改任何既有文件：

1. `project/experiments/architecture_protocol.py` — 实验协议 + 适配器接口
   `BaseArchitectureAdapter` + 有界澄清控制器 `BoundedClarificationController`
   （无 torch 依赖）
2. `project/experiments/run_architecture_experiments.py` — 三组假适配器 + 离线
   运行器 + CLI，输出 `results.json` / `results.csv`
3. `project/tests/test_architecture_experiments.py` — 5 个聚焦测试

## 验收标准逐条对照

| # | 验收标准 | 结果 |
|---|----------|------|
| 1 | 无 torch 也能导入实验模块 | ✅ `torch in sys.modules: False` |
| 2 | fake 模式输出三组 | ✅ monolithic / modular_one_shot / modular_collaborative（2 demo 用例 × 3 组 = 6 行） |
| 3 | 2/3 轮上限稳定，感知失败→证据不足 | ✅ 控制器硬上限；感知异常时收尾并标记 `insufficient` + 风险 |
| 4 | 不改 MVP 默认入口和八项追问 | ✅ 纯新增，未触碰 `assistant_cli.py` / `orchestrator.py` 等 |

## 测试结果

- 聚焦测试 **5/5 通过**
- 全量 67 个测试中 6 个 error，**全部来自既有测试**（`test_experiments` /
  `test_completion_flow` / `test_privacy_logging`），报错为 Windows 上 SQLite
  `sessions.db` 文件占用（`WinError 32`）。本分支仅新增 3 个文件、未改动任何
  既有文件，故与本次改动无关。

## 关键说明

- 代码严格消费负责人冻结的 JSON Schema（组名 `modular_one_shot` 下划线、证据
  `f-N`、决策 `r-N`、`evidence_status`、roadmap 三阶段、元数据与模型输出分离），
  未另建 Pydantic 平行 schema。
- 假适配器与真适配器共用 `BaseArchitectureAdapter` 接口，成员 B 接真模型时
  即插即用。
- `load_protocol` 在 `dataset/research_protocol.json` 缺失时用冻结 model id 兜底，
  保证本分支独立可测。
