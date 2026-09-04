# 成员B 第3周验收记录:固定样例跑通

状态:**已完成(2026-08-23)**。基于 `work/member-b-planning-prompt` 分支
(规划 Prompt `planning-v1`),真实 DeepSeek 调用一次,证据在本地日志
`data/logs/runs.jsonl`(已忽略目录,不提交)。

## 固定样例

| 项 | 值 |
|---|---|
| session_id | `b-week3-check2` |
| 目标 | 获得数据分析实习 |
| 输入文本 | 统计学本科大三，熟悉Python和SQL，做过课程项目 |
| 八项追问 | education=本科大三, major=统计学, skills=Python/SQL, interests=数据分析/机器学习, target_role=数据分析师, time_budget=10小时, preference=杭州, constraints=缺少实习项目经验 |
| 反馈 | 合适(2) |

样例内容为自拟匿名数据,不含任何个人身份信息。

## 手册验收对照

| 手册要求 | 证据 |
|---|---|
| 文字 → 基础规划 | `served_by=cloud_brain`、`model=deepseek-v4-flash`、`status=completed` |
| 方向/差距/阶段行动齐全 | `target_roles` 2 项;`gap_analysis` 3 条;`roadmap_30_90_180` 三阶段齐全 |
| 规划引用检索事实(工作项4) | gap 首条:"…知识库中数据分析师核心技能包含可视化与A/B测试" |
| 画像 → 规划闭环 | 7 个 pipeline 事件:input → perception → follow_up → profile → knowledge → plan → feedback |
| JSONL 日志完整且脱敏 | 记录含 `latency_ms=11445`;扫描无密钥/身份信息 |

## 复现命令

```powershell
printf "2\n" | python -m project.assistant_cli `
  --session-id b-week3-check2 `
  --goal "获得数据分析实习" `
  --text "统计学本科大三，熟悉Python和SQL，做过课程项目" `
  --answers-json '{"education":"本科大三","major":"统计学","skills":"Python, SQL","interests":"数据分析, 机器学习","target_role":"数据分析师","time_budget":"10小时","preference":"杭州","constraints":"缺少实习项目经验"}'
```

## 待转负责人的协作问题(成员A 模块)

CLI 的 JSONL 日志**只在收到反馈后写入**
(`orchestrator.submit_feedback` → `log_run`)。非交互/管道运行若在反馈环节
EOF,该次运行不会留下日志记录。建议:管道演示时显式传入反馈(如 `printf "2\n"`),
或为 CLI 增加 `--feedback` 参数以便实验脚本落日志。

## 下周衔接(M2,第4周)

手册 M2 要求"运行 5 个固定样例且日志完整"。本周已跑通 1 个真实样例 +
离线四组对照 8 行;剩余 4 个真实样例留待第 4 周统一执行,避免重复消耗余额。
