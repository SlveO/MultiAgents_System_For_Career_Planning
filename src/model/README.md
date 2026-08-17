# src/model —— 模型模块（成员B）

DeepSeek 调用、Prompt、画像提取、规划生成与反馈调整。本周交付：客户端 + 适配器 + 画像基线。

## 依赖与环境

- 环境变量：`DEEPSEEK_API_KEY`（必填，密钥绝不写入代码/日志/仓库）
- 依赖：`requests`、`jsonschema`；测试：`pytest`
- 模型：`deepseek-v4-flash`（Base URL https://api.deepseek.com，默认非思考模式）

## 供 A 接入的稳定接口

```python
from src.model.model_adapter import ModelAdapter

adapter = ModelAdapter()  # 自动读取环境变量与 schemas/user_profile.schema.json
result = adapter.extract_profile(raw_input)  # raw_input 为 A 解析后的纯文本
# result = {
#   "profile": {...},          # 七字段画像，必含 missing_fields
#   "errors": [error_dict],    # 失败时非空；格式见下
#   "used_fallback": bool,     # True=API 失败已走规则兜底
#   "elapsed_ms": int,
#   "usage": {...} | None,     # token 用量（供 metrics）
# }
```

`extract_profile` **永不抛异常**；最差返回规则兜底画像（全部字段置空 + missing_fields 齐全）。

## 失败返回格式（写入 SessionState.errors）

| code | 含义 | retryable |
|------|------|-----------|
| MODEL_CONFIG | 缺少 DEEPSEEK_API_KEY 配置 | false |
| MODEL_AUTH | 密钥无效（HTTP 401） | false |
| MODEL_BALANCE | 余额不足（HTTP 402） | false |
| MODEL_RATE_LIMIT | 限流（HTTP 429） | true |
| MODEL_TIMEOUT | 调用超时 | true |
| MODEL_SERVER | 服务端 5xx | true |
| MODEL_HTTP | 其他网络/HTTP 错误 | false |
| MODEL_JSON_ERROR | 输出非合法 JSON 或未过 schema 校验 | false |

每个错误对象：`{"module": "model", "code": "...", "message": "中文可读提示", "retryable": bool}`。

## 模块测试

```bash
.venv/Scripts/python.exe -m pytest tests/test_deepseek_client.py tests/test_model_adapter.py -v
```
