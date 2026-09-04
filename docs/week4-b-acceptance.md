# 成员B 第4周验收记录:M2 五样例 + 规划质量修正

状态:**已完成(2026-08-23)**。基于 `work/member-b-planning-prompt` 分支
(规划 Prompt `planning-v1`)。5 个固定样例全部真实 DeepSeek 调用,
日志在本地 `data/logs/runs.jsonl`(已忽略目录)。

## 五个固定样例结果

| 样例 | 目标 | 画像要点 | served_by | 解析 | 方向 | 差距 | 30/90/180 | 知识库行为 |
|---|---|---|---|---|---|---|---|---|
| b-week3-check2 | 数据分析实习 | 统计学大三,Python/SQL | cloud_brain | ✅ | 2 | 3 | ✅ | 引用事实 |
| b-week4-s1 | 后端开发实习 | 计算机大二,Java/MySQL | cloud_brain | ✅ | 2 | 3 | ✅ | 引用事实 |
| b-week4-s2 | 转行新媒体运营 | 汉语言文学大三,写作 | cloud_brain | ✅ | 2 | 4 | ✅ | 诚实标注未命中 |
| b-week4-s3 | 嵌入式开发 | 电子信息研一,C/单片机 | cloud_brain | ✅ | 2 | 3 | ✅ | 引用事实 |
| b-week4-s4 | 会计师事务所 | 会计大四,初级会计 | cloud_brain | ✅ | 2 | 3 | ✅ | 诚实标注未命中 |

汇总:解析成功 5/5,方向/差距/阶段行动齐全 5/5,每例 7 个 pipeline 事件,
日志无密钥/身份信息,无 fallback(全部 cloud_brain),耗时 7.8-12.7s。

## 规划质量修正结论(手册第4周 B 任务)

对 5 份规划逐条检查后,**规划 Prompt 无需修改**,planning-v1 维持:

1. 知识库命中的 3 例,差距分析全部明确引用知识库事实;
2. 知识库未命中的 2 例(新媒体运营、审计),risk_flags 均按 Prompt 要求 3
   如实标注"知识库未命中/未提供该岗位信息",建议基于画像保守给出,
   **没有任何虚构职业事实**;
3. 未出现 JSON 解析重试或模板降级;
4. 字段完整性:诊断调用(b-week4-diag)确认模型按 schema 完整返回 9 字段,
   含 `learning_resources`(5 项)、`follow_up_questions`(4 项)、
   `confidence=0.85`。

**转成员A 的发现**:`JsonlRunLogger` 的 `output` 记录为 7 字段精简版
(`run_logging.py`),未落 `learning_resources`/`follow_up_questions`/`confidence`。
若结题实验统计需要这三个字段,建议 A 扩展日志 schema;模型侧无需改动。

**转成员C/负责人的发现**:65 条知识库缺少新媒体运营、会计/审计方向岗位,
检索退化为通用岗位(career-001..004)。这是数据覆盖问题(属 C 的
`dataset/career_knowledge_base.json`),建议第 5 周补录或确认覆盖范围。

## M2 验收对照(手册)

| 要求 | 证据 |
|---|---|
| 画像、追问、检索、规划全流程 | 每例 7 事件:input→perception→follow_up→profile→knowledge→plan→feedback |
| 运行 5 个固定样例且日志完整 | 5/5,`data/logs/runs.jsonl` 逐条可查,含 latency/status/feedback |
| 画像→规划闭环 | 5/5 completed,无异常退出 |

## 复现

```powershell
printf "2\n" | python -m project.assistant_cli `
  --session-id b-week4-s1 --goal "获得后端开发实习" `
  --text "计算机科学与技术大二，会Java和MySQL，写过课程管理系统" `
  --answers-json '{"education":"本科大二","major":"计算机科学与技术","skills":"Java, MySQL","interests":"后端开发, 系统设计","target_role":"后端开发工程师","time_budget":"12小时","preference":"深圳","constraints":"没有实习经历"}'
```

其余 4 例同模式,answers-json 见本地日志记录。

## 下周衔接

- 第5周 B 任务:三档输出调整(过简/适中/过详按反馈重新生成)——已确认
  整合版缺失此逻辑,`submit_feedback` 目前只记录不重生成,待第 5 周实现;
- 知识库覆盖缺口(新媒体/审计)待 C 处理;
- M3(第6周)要求 20 例可重复运行,样例清单可复用本表并扩展。
