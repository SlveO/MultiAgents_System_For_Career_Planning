# 第一周模型基线设计（成员B）

日期：2026-08-14
状态：已获用户确认
依据：《多模态职业规划智能助手_团队执行手册（匿名版）》V1.0 成员B任务卡

## 1. 背景与目标

成员B负责"模型如何理解和生成"。第一周目标（手册第1周冲刺：**API与画像基线**）：

1. 跑通 DeepSeek 官方 API（任务1：`src/model/deepseek_client.py`，第1周第2天）
2. 统一适配器：超时、重试、JSON 校验（任务2：`src/model/model_adapter.py`，第1周末）
3. 返回第一个 user_profile JSON，3 个样例稳定返回（48h 行动清单 12-24h / 24-48h）
4. 失败返回格式（48h 行动清单 24-48h）
5. `.env.example` 配置模板，绝不提交真实密钥（第1周立即行动）

## 2. 范围

### 做
- DeepSeek 官方 API 客户端（requests 手写，环境变量密钥）
- 统一适配器：超时、指数退避重试、JSON 解析与 schema 校验
- user_profile JSON Schema（手册冻结字段）
- 画像 Prompt 基线版 + 规则提取兜底
- 结构化失败返回格式（模块、错误码、可读提示）
- pytest 测试（mock，不耗 API 额度）+ 真实 API 验证脚本

### 不做（后续周次或他人职责）
- 规划生成 Prompt（第3周）、三档反馈调整（第5周）、fallback 模板模块（第6周）
- 会话日志基础设施（A 的 `src/logging/`）；B 只用标准 `logging` 输出，后续由 A 接入
- 终端输入、文件上传、知识库（A/C 职责）
- 画像 20 样例 ≥90% 成功率达标（第2周任务3的验收，本周只做基线）

## 3. 文件结构

```
src/model/
  __init__.py
  deepseek_client.py   # 底层API调用
  model_adapter.py     # 统一适配器
  prompts/
    __init__.py
    profile.py         # 画像Prompt基线版
schemas/
  user_profile.schema.json
tests/
  test_model_adapter.py
  fixtures/profile_samples.json   # 3个固定样例
scripts/
  demo_profile.py      # 真实API验证脚本
.env.example           # 配置模板（无真实密钥）
.env                   # 本地真实密钥（gitignore）
.gitignore             # .env、111.txt、__pycache__、.venv、outputs等
requirements.txt       # requests、jsonschema、pytest
outputs/week1/         # 运行时生成的验证证据（gitignore）
```

## 4. 组件设计

### 4.1 `deepseek_client.py` — `DeepSeekClient`

- 构造：`DeepSeekClient()`；密钥从环境变量 `DEEPSEEK_API_KEY` 读取，缺失时抛 `ModelConfigError`（中文提示如何配置）
- 常量：`BASE_URL = "https://api.deepseek.com"`，`DEFAULT_MODEL = "deepseek-v4-flash"`
- 模型模式：不传思考/推理相关参数（手册要求默认非思考模式，保证速度和JSON稳定）
- `list_models()` → GET `/models`，返回模型 id 列表（手册要求调用前确认模型可用）
- `chat(messages, *, json_mode=True, temperature=0.3, max_tokens=1500, timeout=30)` → `(content: str, usage: dict)`
  - `json_mode=True` 时附带 `response_format={"type": "json_object"}`
  - usage 含 prompt/completion/total tokens，供 A 的 metrics 汇总
- 错误映射（HTTP 与网络异常 → 类型化异常）：
  - 401 → `ModelAuthError`（密钥无效）
  - 402 → `ModelBalanceError`（余额不足）
  - 429 → `ModelRateLimitError`（限流）
  - 5xx → `ModelServerError`
  - 连接/读取超时 → `ModelTimeoutError`
  - 其他 → `ModelHTTPError`
- 密钥不出现在任何日志、异常信息、返回值中

### 4.2 `model_adapter.py` — `ModelAdapter`

- 统一入口：`generate_json(system_prompt, user_content, json_schema) -> dict`
  - 调用 client.chat，`json_mode=True`
  - 解析：先尝试 `json.loads`；失败则剥离 ```json 围栏后再试
  - schema 校验：`jsonschema` 库校验；校验失败视为 JSON 错误
- 重试策略（与手册3.2"重试一次；仍失败则使用规则提取"严格对齐）：
  - 任何失败最多重试 1 次（总尝试 2 次）：网络类错误（超时/429/5xx）等待 1 秒后重试；JSON 错误追加"必须只输出合法JSON"强化约束后重试
  - 重试后仍失败：`generate_json` 抛出对应类型化异常，由调用方决定降级；`extract_profile` 捕获异常后走规则兜底
- `extract_profile(raw_input) -> dict`，返回 `{"profile": {...}, "errors": [...], "used_fallback": bool}`
  - 流程：API 提取 → 失败重试 → 仍失败走 `_rule_based_profile()` 规则兜底
  - 规则兜底：关键词/正则简单提取（专业、年级等），提取不到的全部字段置空并写入 `missing_fields`；绝不虚构
  - 永不抛裸异常；所有失败记录进 errors
- 失败返回格式（供 A 写入 SessionState.errors）：
  ```json
  {"module": "model", "code": "MODEL_TIMEOUT", "message": "DeepSeek接口超时", "retryable": true}
  ```
  错误码全集：`MODEL_CONFIG`、`MODEL_AUTH`、`MODEL_BALANCE`、`MODEL_RATE_LIMIT`、`MODEL_TIMEOUT`、`MODEL_SERVER`、`MODEL_HTTP`、`MODEL_JSON_ERROR`
- 日志：标准 `logging.getLogger("model")`，记录耗时、重试次数、错误码，不记录输入原文密钥

### 4.3 `schemas/user_profile.schema.json` — 手册冻结字段

| 字段 | 类型 | 说明 |
|------|------|------|
| major | string | 专业 |
| grade | string | 年级 |
| skills | array[string] | 技能 |
| interests | array[string] | 兴趣 |
| career_goals | array[string] | 职业目标 |
| output_preference | string (enum: too_short/suitable/too_detailed) | 输出偏好 |
| missing_fields | array[string]（必填） | 缺失字段 |

所有字段允许空值（信息不足时置空并写入 missing_fields），`missing_fields` 必填且必须为数组。schema 使用 draft-07，`additionalProperties: false`。

### 4.4 `prompts/profile.py` — 画像 Prompt 基线版

- system prompt 规则：
  1. 只提取用户明确陈述的信息，禁止臆造、禁止补充常识推断
  2. 无法确定的字段必须出现在 `missing_fields` 中
  3. 只输出符合 schema 的合法 JSON，不输出解释文字
  4. 输出偏好未提及时置空并写入 `missing_fields`（不替用户猜测，留给C的追问规则补充）

### 4.5 `scripts/demo_profile.py` — 真实 API 验证

- 从 `tests/fixtures/profile_samples.json` 读 3 个样例，逐个调 `extract_profile`
- 输出每个样例的画像 JSON、耗时、token 用量、errors；结果存 `outputs/week1/`
- 环境变量缺失时从 `.env` 读取（仅脚本层便利，客户端本身只读环境变量）

## 5. 安全约定

- 真实密钥只存在于：`111.txt`（用户本地原始文件）与 `.env`（本地配置），两者均进 `.gitignore`
- 代码、日志、测试、示例中不含任何密钥；`.env.example` 只含占位符
- 本工作区目前无 git 仓库；加入团队仓库前需确认 `.gitignore` 生效

## 6. 测试与验证

### 单元测试（pytest + unittest.mock，不依赖网络与密钥）
1. 密钥缺失 → 清晰配置错误
2. 429/402/401/超时 → 对应类型化异常（mock HTTP 响应）
3. 超时重试：首次超时、第二次成功 → 返回数据且重试计数正确
4. JSON 恢复：```json 围栏包裹 → 解析成功
5. schema 校验：非法结构 → 触发 JSON 重试；再失败 → 兜底
6. 规则兜底：API 全部失败 → 返回规则画像 + missing_fields + errors 非空
7. 画像"不得虚构"：本周由两层保证——(a) Prompt 硬约束（API 路径）；(b) 规则兜底路径的测试断言：输入不含任何可提取信息时，规则画像全部字段置空且 `missing_fields` 包含全部字段，输出中不出现输入未提及的内容。API 路径不做后置过滤（第2周画像达标时再评估是否需要）

### 真实验证（用户已授权，消耗少量余额）
1. `/models` 确认 `deepseek-v4-flash` 可用
2. 3 个样例稳定返回合法 user_profile JSON（schema 校验通过）
3. 截图/保存输出到 `outputs/week1/` 作为交接证据

## 7. 验收对照（手册成员B验收标准）

| 手册标准 | 本周落实 |
|---------|---------|
| 密钥只存在 DEEPSEEK_API_KEY 环境变量 | 客户端仅读环境变量；.env/.gitignore 双保险 |
| 20 样例 JSON 解析成功率 ≥90% | 本周基线：3 样例跑通；20 样例达标为第2周任务 |
| 不得虚构、缺失写 missing_fields | Prompt 硬约束 + 规则兜底 + schema 必填 missing_fields |
| API 超时/限流/余额/JSON 错误有明确提示和日志 | 类型化异常 + 错误码 + logging |

## 8. 依赖

- Python ≥3.11（本机 3.13.9 开发；代码保持 3.11 兼容语法，与团队统一环境一致）
- `requests`（HTTP）、`jsonschema`（schema 校验）、`pytest`（测试）
- 虚拟环境：`.venv`（gitignore）
