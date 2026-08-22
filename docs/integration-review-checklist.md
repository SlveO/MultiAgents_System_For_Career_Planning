# 成员分支整合审查清单

## 审查记录

每个成员分支单独填写：负责人、分支、提交哈希、交付物、依赖或阻塞、
验收命令、结果、决定（接受/修改后再审/阻塞）和下一步动作。不得只写
“已完成”。

## 通用检查

1. 确认分支基于 `origin/integration/week1-results`，且只包含负责人分配的
   文件范围。
2. 阅读完整差异，拒绝平行 `src/`、强制 GPU 导入、模型权重、运行日志、
   缓存、原始上传、密钥、私人路径和身份信息。
3. 检查新增 JSON、Markdown、命令和依赖是否与
   `research_*.schema.json`、`research_protocol.json`、评分量表和
   `research-decisions.md` 一致；成员不得绕过严格解析或自行更改提示词。
4. 运行 `git diff --check`、`python -m compileall -q project`、针对性测试和
   `python -m unittest discover -s project/tests -v`。
5. 将通过、跳过、依赖阻塞和真实失败分开记录；不得把 Fake 输出写成模型
   质量结果。

只读审查命令示例：

```powershell
git fetch origin
git log --oneline origin/integration/week1-results..origin/work/<role>-<task>
git diff --stat origin/integration/week1-results...origin/work/<role>-<task>
git diff --check origin/integration/week1-results...origin/work/<role>-<task>
```

## 分工验收

### 成员 A

- 没有 `torch` 也能导入实验模块；
- Fake 模式输出单体、一次性模块化和协作三组；
- 2/3 轮上限稳定，感知失败后最终规划含证据不足风险；
- 不修改 MVP 默认入口和八项用户追问。

### 成员 B

- 下载和推理只在 L20 执行，脚本不含账号、主机、令牌或私人绝对路径；
- 模型 ID、修订、Ubuntu/驱动/CUDA/PyTorch 和峰值显存记录完整；
- 三个模型各有一次确定性冒烟证据，失败时保留准确错误；
- 不提交权重或 `data/` 运行结果。

### 成员 C

- 新增职业来源 URL 可访问且与记录内容相关；
- 6 个试验案例为 3 个图像和 3 个 PDF 页面，使用许可清楚；
- 每例有固定画像、固定知识片段、素材哈希和 3–5 条预期证据；
- 决定性证据不在文本重复，素材和元数据不含身份信息。

## 整合决定模板

```text
负责人：
分支与提交：
交付物：
依赖/阻塞：
已运行命令及结果：
验收项：通过 / 修改后再审 / 阻塞
主要问题：
允许整合的文件：
下一步动作与期限：
```

负责人只整合明确通过的文件。发生范围外改动、不可复现结果或隐私问题时，
先退回修改，不通过合并顺手修复。

## 2026-08-22 负责人返工记录

- 成员 A：选择性移植实验运行器后，由负责人补齐严格 Schema、素材哈希、
  案例驱动 Fake 输出、2/3轮差异、错误代码脱敏和外部适配器注入测试；本地
  测试通过，待最终差异审阅。
- 成员 B：确认原交付未在 L20 执行，不整合为已完成成果。返工边界和两个
  检查点见 `docs/member-b-l20-agent-handoff.md`。
- 成员 C：原20条岗位记录仅作为来源线索保留；负责人修正空链接与错字，并
  新建3图像+3 PDF匿名自制案例。来源线索不再标记为正式字段真值。

本轮没有合并、commit 或 push；最终决定仍需负责人查看差异和测试报告。
