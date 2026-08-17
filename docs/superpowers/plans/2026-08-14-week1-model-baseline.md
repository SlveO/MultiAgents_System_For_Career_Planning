# 第一周模型基线（成员B）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通 DeepSeek 官方 API，交付带超时/重试/JSON 校验的统一模型适配器和画像提取基线（含 3 个样例真实验证）。

**Architecture:** 两层结构——`DeepSeekClient` 只负责 HTTP 调用与错误映射；`ModelAdapter` 在其上实现重试、JSON 解析校验与规则兜底，对 A 的编排器暴露 `generate_json` / `extract_profile` 两个稳定入口。画像字段由 `schemas/user_profile.schema.json` 冻结，Prompt 硬约束"不得虚构"。

**Tech Stack:** Python 3.11 兼容语法（本机 3.13.9）、requests、jsonschema、pytest、DeepSeek API（deepseek-v4-flash）。

**设计依据:** `docs/superpowers/specs/2026-08-14-week1-model-baseline-design.md`

---

## 前置说明

- 所有命令在 `d:\member_B`（项目根目录）下、Git Bash 中执行。
- 测试运行用 `.venv/Scripts/python.exe -m pytest`（Windows 虚拟环境路径）。
- 本工作区无团队 git 仓库，Task 0 会初始化**本地个人仓库**（便于回滚；不影响团队仓库，B 的代码最终拷贝进团队仓库的 feature/b-* 分支）。
- 真实密钥只写入 `.env`（gitignore），本计划任何命令都不回显密钥内容。

---

### Task 0: 环境与仓库初始化

**Files:**
- Create: `.gitignore`、`.env.example`、`.env`、`requirements.txt`、`pytest.ini`、`src/__init__.py`、`src/model/__init__.py`、`src/model/prompts/__init__.py`

- [ ] **Step 1: 创建虚拟环境并安装依赖**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip requests jsonschema pytest
```

Expected: 安装成功，无报错。

- [ ] **Step 2: 创建 .gitignore**

```gitignore
# 密钥与本地配置（绝不入库）
.env
111.txt

# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# 运行时输出
outputs/
```

- [ ] **Step 3: 创建 .env.example（仅模板，不含真实密钥）**

```
# DeepSeek API 配置模板：复制本文件为 .env 并填入真实密钥（.env 不入库）
DEEPSEEK_API_KEY=sk-在这里填入你的密钥
```

- [ ] **Step 4: 生成 .env（真实密钥来自 111.txt，不回显内容）**

```bash
printf 'DEEPSEEK_API_KEY=%s\n' "$(tr -d '\r\n ' < 111.txt)" > .env
grep -c '^DEEPSEEK_API_KEY=sk-' .env
```

Expected: 输出 `1`（确认已写入且格式正确，密钥内容不显示）。

- [ ] **Step 5: 创建 requirements.txt**

```
requests>=2.31
jsonschema>=4.20
pytest>=8.0
```

- [ ] **Step 6: 创建 pytest.ini**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 7: 创建包初始化文件**

```bash
mkdir -p src/model/prompts schemas tests/fixtures scripts outputs
touch src/__init__.py src/model/__init__.py src/model/prompts/__init__.py
```

- [ ] **Step 8: 初始化本地 git 仓库并提交**

```bash
git init
git config user.name "member-B"
git config user.email "member-b@local"
git add .gitignore .env.example requirements.txt pytest.ini src
git commit -m "chore: 初始化成员B工作区（venv、gitignore、依赖配置）"
```

Expected: 提交成功。验证 `.env` 与 `111.txt` 未被提交：`git status` 显示它们仅出现在 .gitignore 规则中（`git ls-files | grep -E '\.env$|111'` 无输出）。

---

### Task 1: DeepSeekClient（底层 HTTP 封装 + 类型化错误）

**Files:**
- Create: `src/model/deepseek_client.py`
- Test: `tests/test_deepseek_client.py`

- [ ] **Step 1: 编写失败测试**

`tests/test_deepseek_client.py`：

```python
"""DeepSeekClient 单元测试（mock HTTP，不依赖网络与密钥）。"""
import pytest
import requests

from src.model.deepseek_client import (
    DeepSeekClient,
    ModelAuthError,
    ModelBalanceError,
    ModelConfigError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"error": {"message": "测试错误"}}

    def json(self):
        return self._body

    @property
    def text(self):
        return str(self._body)


def make_client(monkeypatch, api_key="sk-test"):
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    return DeepSeekClient()


class TestConfig:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ModelConfigError) as exc:
            DeepSeekClient()
        assert exc.value.code == "MODEL_CONFIG"

    def test_key_from_env_used(self, monkeypatch):
        c = make_client(monkeypatch, "sk-env-key")
        assert c.api_key == "sk-env-key"
        assert "sk-env-key" not in repr(c)


class TestErrorMapping:
    @pytest.mark.parametrize(
        "status,error_cls",
        [
            (401, ModelAuthError),
            (402, ModelBalanceError),
            (429, ModelRateLimitError),
            (500, ModelServerError),
            (418, ModelHTTPError),
        ],
    )
    def test_status_maps_to_typed_error(self, monkeypatch, status, error_cls):
        c = make_client(monkeypatch)
        with pytest.raises(error_cls):
            c._raise_for_status(FakeResponse(status))

    def test_error_dict_format(self, monkeypatch):
        c = make_client(monkeypatch)
        with pytest.raises(ModelRateLimitError) as exc:
            c._raise_for_status(FakeResponse(429))
        err = exc.value.to_error_dict()
        assert err["module"] == "model"
        assert err["code"] == "MODEL_RATE_LIMIT"
        assert err["retryable"] is True
        assert "限流" in err["message"]


class TestChat:
    def test_chat_success_and_payload(self, monkeypatch):
        c = make_client(monkeypatch)
        fake = FakeResponse(200, {
            "choices": [{"message": {"content": '{"major": "计算机"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        sent = {}

        def fake_post(url, **kwargs):
            sent["url"] = url
            sent.update(kwargs)
            return fake

        monkeypatch.setattr(c._session, "post", fake_post)
        content, usage = c.chat([{"role": "user", "content": "输出json"}])
        assert content == '{"major": "计算机"}'
        assert usage["total_tokens"] == 15
        assert sent["url"].endswith("/chat/completions")
        assert sent["json"]["model"] == "deepseek-v4-flash"
        assert sent["json"]["response_format"] == {"type": "json_object"}

    def test_chat_timeout_maps_to_timeout_error(self, monkeypatch):
        c = make_client(monkeypatch)

        def fake_post(*a, **kw):
            raise requests.exceptions.Timeout("boom")

        monkeypatch.setattr(c._session, "post", fake_post)
        with pytest.raises(ModelTimeoutError) as exc:
            c.chat([{"role": "user", "content": "x"}])
        assert exc.value.retryable is True

    def test_chat_network_error_maps_to_http_error(self, monkeypatch):
        c = make_client(monkeypatch)

        def fake_post(*a, **kw):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(c._session, "post", fake_post)
        with pytest.raises(ModelHTTPError):
            c.chat([{"role": "user", "content": "x"}])


class TestListModels:
    def test_list_models(self, monkeypatch):
        c = make_client(monkeypatch)
        fake = FakeResponse(200, {"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-v4-pro"}]})
        monkeypatch.setattr(c._session, "get", lambda *a, **kw: fake)
        assert c.list_models() == ["deepseek-v4-flash", "deepseek-v4-pro"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_deepseek_client.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.model.deepseek_client'`

- [ ] **Step 3: 实现 `src/model/deepseek_client.py`**

```python
"""DeepSeek 官方 API 底层客户端。

职责边界（成员B）：
- 只负责 HTTP 调用、超时与错误映射；
- 不做重试（适配器负责）、不做 JSON 校验（适配器负责）；
- 密钥仅从环境变量 DEEPSEEK_API_KEY 读取，绝不写入日志。

本文件同时定义模型错误体系：所有错误继承 ModelError，
可用 to_error_dict() 转为 A 的 SessionState.errors 条目格式。
"""

import logging
import os

import requests

logger = logging.getLogger("model")

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class ModelError(Exception):
    """模型调用错误基类：module/code/message/retryable 四元组。"""

    module = "model"
    code = "MODEL_ERROR"
    retryable = False

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def to_error_dict(self):
        """转为 SessionState.errors 条目格式。"""
        return {
            "module": self.module,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class ModelConfigError(ModelError):
    code = "MODEL_CONFIG"


class ModelAuthError(ModelError):
    code = "MODEL_AUTH"


class ModelBalanceError(ModelError):
    code = "MODEL_BALANCE"


class ModelRateLimitError(ModelError):
    code = "MODEL_RATE_LIMIT"
    retryable = True


class ModelTimeoutError(ModelError):
    code = "MODEL_TIMEOUT"
    retryable = True


class ModelServerError(ModelError):
    code = "MODEL_SERVER"
    retryable = True


class ModelHTTPError(ModelError):
    code = "MODEL_HTTP"


class ModelJSONError(ModelError):
    """模型输出不是合法 JSON 或未通过 schema 校验（适配器抛出）。"""

    code = "MODEL_JSON_ERROR"


_STATUS_TO_ERROR = {
    401: ModelAuthError,
    402: ModelBalanceError,
    429: ModelRateLimitError,
}

_STATUS_MESSAGE = {
    401: "API 密钥无效，请检查 DEEPSEEK_API_KEY",
    402: "账户余额不足，请到 DeepSeek 平台充值后重试",
    429: "请求被限流，请稍后重试",
}


class DeepSeekClient:
    """DeepSeek API 客户端（requests 手写，非思考模式）。"""

    def __init__(self, api_key=None, base_url=BASE_URL, model=DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ModelConfigError(
                "未找到 DEEPSEEK_API_KEY 环境变量：请在 shell 中 export 该变量，"
                "或复制 .env.example 为 .env 并填入真实密钥（.env 不入库）"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def __repr__(self):
        # 任何情况下不泄露密钥
        return f"DeepSeekClient(model={self.model!r}, base_url={self.base_url!r})"

    @staticmethod
    def _raise_for_status(resp):
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        cls = _STATUS_TO_ERROR.get(resp.status_code)
        if cls is None:
            cls = ModelServerError if resp.status_code >= 500 else ModelHTTPError
        message = _STATUS_MESSAGE.get(resp.status_code, f"HTTP {resp.status_code} 错误")
        if detail:
            message += f"（服务端信息：{detail}）"
        raise cls(message)

    def list_models(self, timeout=15):
        """调用 GET /models 返回模型 id 列表（手册要求调用前确认模型可用）。"""
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise ModelTimeoutError(f"获取模型列表超时（>{timeout}s）") from exc
        except requests.exceptions.RequestException as exc:
            raise ModelHTTPError(f"获取模型列表网络错误：{exc.__class__.__name__}") from exc
        self._raise_for_status(resp)
        return [m.get("id", "") for m in resp.json().get("data", [])]

    def chat(self, messages, json_mode=True, temperature=0.3, max_tokens=1500, timeout=30):
        """调用 /chat/completions，返回 (content, usage)。

        json_mode=True 时使用 response_format={"type": "json_object"}
        （DeepSeek 要求此时 prompt 中必须出现 "json" 字样，由调用方保证）。
        不传思考/推理相关参数（手册要求默认非思考模式）。
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions", json=payload, timeout=timeout
            )
        except requests.exceptions.Timeout as exc:
            raise ModelTimeoutError(f"模型调用超时（>{timeout}s）") from exc
        except requests.exceptions.RequestException as exc:
            raise ModelHTTPError(f"模型调用网络错误：{exc.__class__.__name__}") from exc
        self._raise_for_status(resp)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, usage
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_deepseek_client.py -v`
Expected: PASS（12 passed）

- [ ] **Step 5: 提交**

```bash
git add src/model/deepseek_client.py tests/test_deepseek_client.py
git commit -m "feat(model): DeepSeek API 底层客户端与类型化错误体系（TDD）"
```

---

### Task 2: 画像 Schema 与画像 Prompt（基线版）

**Files:**
- Create: `schemas/user_profile.schema.json`
- Create: `src/model/prompts/profile.py`
- Test: `tests/test_schema_and_prompt.py`

- [ ] **Step 1: 编写失败测试**

`tests/test_schema_and_prompt.py`：

```python
"""画像 schema 与画像 Prompt 的静态校验测试。"""
import json
from pathlib import Path

import jsonschema
import pytest

from src.model.prompts.profile import PROFILE_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "user_profile.schema.json"


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class TestSchema:
    def test_schema_loads_and_validates_good_profile(self):
        schema = load_schema()
        good = {
            "major": "计算机科学与技术",
            "grade": "大三",
            "skills": ["Python", "C++"],
            "interests": ["人工智能"],
            "career_goals": ["算法工程师"],
            "output_preference": "too_detailed",
            "missing_fields": [],
        }
        jsonschema.validate(good, schema)

    def test_schema_rejects_extra_fields(self):
        schema = load_schema()
        bad = {"major": "计算机", "missing_fields": [], "name": "张三"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_schema_requires_missing_fields(self):
        schema = load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"major": "计算机"}, schema)

    def test_schema_allows_empty_output_preference(self):
        schema = load_schema()
        jsonschema.validate({"output_preference": "", "missing_fields": ["major"]}, schema)


class TestPrompt:
    def test_prompt_contains_json_word(self):
        # DeepSeek JSON 模式要求 prompt 中出现 "json" 字样
        assert "JSON" in PROFILE_SYSTEM_PROMPT

    def test_prompt_forbids_fabrication(self):
        assert "禁止臆造" in PROFILE_SYSTEM_PROMPT
        assert "missing_fields" in PROFILE_SYSTEM_PROMPT
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schema_and_prompt.py -v`
Expected: FAIL（schema 文件不存在 / profile 模块不存在）

- [ ] **Step 3: 创建 `schemas/user_profile.schema.json`**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "user_profile",
  "description": "成员B定义的画像字段（手册冻结）：专业、年级、技能、兴趣、职业目标、输出偏好、缺失字段。所有字段允许空值；信息不足时置空并写入 missing_fields。",
  "type": "object",
  "additionalProperties": false,
  "required": ["missing_fields"],
  "properties": {
    "major": {"type": "string", "description": "专业"},
    "grade": {"type": "string", "description": "年级"},
    "skills": {"type": "array", "items": {"type": "string"}, "description": "技能"},
    "interests": {"type": "array", "items": {"type": "string"}, "description": "兴趣"},
    "career_goals": {"type": "array", "items": {"type": "string"}, "description": "职业目标"},
    "output_preference": {
      "type": "string",
      "enum": ["", "too_short", "suitable", "too_detailed"],
      "description": "输出偏好（空字符串=用户未提及）"
    },
    "missing_fields": {
      "type": "array",
      "items": {"type": "string"},
      "description": "用户未提供信息的字段名数组（必填）"
    }
  }
}
```

- [ ] **Step 4: 创建 `src/model/prompts/profile.py`**

```python
"""用户画像提取 Prompt（基线版，第1周）。"""

PROFILE_SYSTEM_PROMPT = """你是一个职业规划助手的信息提取模块。你的任务是从用户的输入中提取用户画像，输出 JSON。

严格规则：
1. 只提取用户在输入中明确陈述的信息，禁止臆造、禁止推断、禁止补充任何用户未提供的内容。
2. 字段说明：
   - major: 专业（字符串，用户未提及时为空字符串）
   - grade: 年级（字符串，用户未提及时为空字符串）
   - skills: 技能（字符串数组，用户未提及时为空数组）
   - interests: 兴趣（字符串数组，用户未提及时为空数组）
   - career_goals: 职业目标（字符串数组，用户未提及时为空数组）
   - output_preference: 输出偏好，取值 too_short / suitable / too_detailed，用户未提及输出详略偏好时为空字符串
   - missing_fields: 用户未提供信息的字段名数组（必填，例如 ["grade", "skills"]）
3. 某个字段无法从用户输入确定时，该字段置空（字符串置 ""，数组置 []），并把字段名加入 missing_fields。
4. 只输出一个符合上述结构的合法 JSON 对象，不要输出任何解释文字、Markdown 代码围栏或其他内容。
"""
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_schema_and_prompt.py -v`
Expected: PASS（6 passed）

- [ ] **Step 6: 提交**

```bash
git add schemas/user_profile.schema.json src/model/prompts/profile.py tests/test_schema_and_prompt.py
git commit -m "feat(model): user_profile Schema 与画像 Prompt 基线（TDD）"
```

---

### Task 3: ModelAdapter —— generate_json（JSON 解析 + 重试 + 校验）

**Files:**
- Create: `src/model/model_adapter.py`
- Test: `tests/test_model_adapter.py`（本任务先写 generate_json 相关测试类）

- [ ] **Step 1: 编写失败测试**

`tests/test_model_adapter.py`：

```python
"""ModelAdapter 单元测试（FakeClient 桩，不依赖网络与密钥）。"""
import json

import pytest

from src.model.deepseek_client import (
    ModelBalanceError,
    ModelJSONError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from src.model.model_adapter import ModelAdapter


class FakeClient:
    """按序返回预设响应或抛异常的客户端桩。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


VALID_PROFILE = {
    "major": "计算机科学与技术",
    "grade": "大三",
    "skills": ["Python"],
    "interests": ["人工智能"],
    "career_goals": ["算法工程师"],
    "output_preference": "too_detailed",
    "missing_fields": [],
}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class TestGenerateJson:
    def test_success(self):
        adapter = ModelAdapter(client=FakeClient([(json.dumps(VALID_PROFILE, ensure_ascii=False), USAGE)]))
        data = adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert data == VALID_PROFILE

    def test_retry_once_on_timeout_then_success(self):
        client = FakeClient([
            ModelTimeoutError("超时"),
            (json.dumps(VALID_PROFILE, ensure_ascii=False), USAGE),
        ])
        adapter = ModelAdapter(client=client)
        data = adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert data == VALID_PROFILE
        assert len(client.calls) == 2

    def test_no_retry_on_balance_error(self):
        client = FakeClient([ModelBalanceError("余额不足")])
        adapter = ModelAdapter(client=client)
        with pytest.raises(ModelBalanceError):
            adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert len(client.calls) == 1

    def test_json_fence_wrapped_content_parses(self):
        content = "```json\n" + json.dumps(VALID_PROFILE, ensure_ascii=False) + "\n```"
        adapter = ModelAdapter(client=FakeClient([(content, USAGE)]))
        data = adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert data == VALID_PROFILE

    def test_invalid_json_retries_with_reinforced_constraint(self):
        client = FakeClient([
            ("这不是JSON", USAGE),
            (json.dumps(VALID_PROFILE, ensure_ascii=False), USAGE),
        ])
        adapter = ModelAdapter(client=client)
        data = adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert data == VALID_PROFILE
        assert len(client.calls) == 2
        last_message = client.calls[1][-1]["content"]
        assert "合法 JSON" in last_message

    def test_schema_violation_retries_then_success(self):
        bad = dict(VALID_PROFILE, name="张三")  # additionalProperties 违规
        client = FakeClient([
            (json.dumps(bad, ensure_ascii=False), USAGE),
            (json.dumps(VALID_PROFILE, ensure_ascii=False), USAGE),
        ])
        adapter = ModelAdapter(client=client)
        data = adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert data == VALID_PROFILE
        assert len(client.calls) == 2

    def test_all_attempts_fail_raises_json_error(self):
        client = FakeClient([("不是JSON", USAGE), ("还不是JSON", USAGE)])
        adapter = ModelAdapter(client=client)
        with pytest.raises(ModelJSONError):
            adapter.generate_json("系统", "用户输入", adapter.profile_schema)
        assert len(client.calls) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_adapter.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'src.model.model_adapter'`（fixtures 文件缺失会在实现后另测，本任务测试不读 samples 文件）

- [ ] **Step 3: 实现 `src/model/model_adapter.py`（generate_json 部分 + 公共工具）**

```python
"""统一模型适配器：超时/重试/JSON 校验与结构化失败格式。

对上层（A 的编排器）暴露稳定接口：
- generate_json(system_prompt, user_content, json_schema) -> dict
- extract_profile(raw_input) -> {"profile", "errors", "used_fallback", "elapsed_ms", "usage"}

失败返回格式（供 A 写入 SessionState.errors）：
{"module": "model", "code": "MODEL_XXX", "message": "中文提示", "retryable": bool}

重试策略（手册 3.2 "重试一次；仍失败则使用规则提取"）：
任何失败最多重试 1 次（总尝试 2 次）；网络类错误（超时/429/5xx）等待 1 秒后重试，
JSON 错误追加强化约束后重试。认证/余额类错误不重试，直接抛出。
"""

import json
import logging
import re
import time
from pathlib import Path

import jsonschema

from src.model.deepseek_client import (
    DeepSeekClient,
    ModelError,
    ModelJSONError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
)
from src.model.prompts.profile import PROFILE_SYSTEM_PROMPT

logger = logging.getLogger("model")

MAX_ATTEMPTS = 2  # 首次 + 1 次重试
RETRY_WAIT_SECONDS = 1
REINFORCE_MESSAGE = "请严格只输出符合要求的合法 JSON，不要输出任何解释文字或 Markdown 代码围栏。"

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "user_profile.schema.json"
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def load_user_profile_schema():
    """加载画像 JSON Schema（文件缺失时给出清晰错误）。"""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"画像 schema 不存在：{_SCHEMA_PATH}")
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def parse_json(content):
    """解析模型输出为 dict；支持 ```json 围栏；失败抛 ModelJSONError。"""
    text = (content or "").strip()
    candidates = [text]
    match = _JSON_FENCE_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise ModelJSONError("模型输出不是合法 JSON")


def _validate_or_raise(data, schema):
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        raise ModelJSONError(f"模型输出未通过 schema 校验：{exc.message}") from exc


class ModelAdapter:
    """统一模型适配器：负责重试、JSON 解析校验、画像兜底。"""

    def __init__(self, client=None, profile_schema=None):
        self.client = client or DeepSeekClient()
        self.profile_schema = profile_schema or load_user_profile_schema()
        self.last_usage = None

    def generate_json(self, system_prompt, user_content, json_schema):
        """通用 JSON 生成：最多尝试 2 次（重试 1 次），最终失败抛类型化异常。

        返回 dict（已通过 json_schema 校验）；成功调用的 usage 记录在 self.last_usage。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        last_error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                content, usage = self.client.chat(messages, json_mode=True)
                data = parse_json(content)
                _validate_or_raise(data, json_schema)
                self.last_usage = usage
                logger.info(
                    "generate_json 成功 attempt=%d total_tokens=%s",
                    attempt, usage.get("total_tokens"),
                )
                return data
            except (ModelTimeoutError, ModelRateLimitError, ModelServerError) as exc:
                last_error = exc
                logger.warning("generate_json 网络类失败 attempt=%d code=%s", attempt, exc.code)
            except ModelJSONError as exc:
                last_error = exc
                logger.warning("generate_json JSON 失败 attempt=%d", attempt)
                messages.append({"role": "user", "content": REINFORCE_MESSAGE})
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
        raise last_error
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_adapter.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add src/model/model_adapter.py tests/test_model_adapter.py
git commit -m "feat(model): 统一适配器 generate_json（重试/JSON解析/校验，TDD）"
```

---

### Task 4: ModelAdapter —— extract_profile 与规则兜底

**Files:**
- Modify: `src/model/model_adapter.py`（追加 extract_profile / _coerce_profile / _rule_based_profile）
- Modify: `tests/test_model_adapter.py`（追加测试类）
- Create: `tests/fixtures/profile_samples.json`

- [ ] **Step 1: 追加失败测试（追加到 tests/test_model_adapter.py 末尾）**

```python
class TestExtractProfile:
    def test_success_normalizes_and_reports_usage(self):
        client = FakeClient([(json.dumps(VALID_PROFILE, ensure_ascii=False), USAGE)])
        adapter = ModelAdapter(client=client)
        result = adapter.extract_profile("我是计算机大三学生，会Python")
        assert result["used_fallback"] is False
        assert result["errors"] == []
        assert result["profile"] == VALID_PROFILE
        assert result["usage"]["total_tokens"] == 15

    def test_api_failure_falls_back_to_rules(self):
        client = FakeClient([ModelTimeoutError("超时"), ModelTimeoutError("超时")])
        adapter = ModelAdapter(client=client)
        result = adapter.extract_profile("我是软件工程专业的大二学生，会Java")
        assert result["used_fallback"] is True
        assert result["profile"]["major"] == "软件工程"
        assert result["profile"]["grade"] == "大二"
        assert result["profile"]["skills"] == ["Java"]
        assert len(result["errors"]) == 1
        assert result["errors"][0]["code"] == "MODEL_TIMEOUT"
        assert result["errors"][0]["retryable"] is True

    def test_fallback_never_fabricates(self):
        client = FakeClient([ModelTimeoutError("超时"), ModelTimeoutError("超时")])
        adapter = ModelAdapter(client=client)
        result = adapter.extract_profile("帮我规划一下")
        profile = result["profile"]
        assert profile["major"] == ""
        assert profile["grade"] == ""
        assert profile["skills"] == []
        assert profile["career_goals"] == []
        assert profile["output_preference"] == ""
        assert set(profile["missing_fields"]) == {
            "major", "grade", "skills", "interests", "career_goals", "output_preference",
        }

    def test_never_raises_on_any_model_error(self):
        client = FakeClient([ModelBalanceError("余额不足"), ModelBalanceError("余额不足")])
        adapter = ModelAdapter(client=client)
        result = adapter.extract_profile("随便说点什么")
        assert result["used_fallback"] is True
        assert result["errors"][0]["code"] == "MODEL_BALANCE"

    def test_coerce_fills_omitted_fields_into_missing(self):
        minimal = {"major": "数学", "missing_fields": ["grade"]}
        client = FakeClient([(json.dumps(minimal, ensure_ascii=False), USAGE)])
        adapter = ModelAdapter(client=client)
        result = adapter.extract_profile("数学专业")
        assert result["used_fallback"] is False
        profile = result["profile"]
        assert profile["major"] == "数学"
        assert profile["skills"] == []
        assert {"grade", "skills", "interests", "career_goals", "output_preference"} <= set(
            profile["missing_fields"]
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_adapter.py -v`
Expected: FAIL，`AttributeError: 'ModelAdapter' object has no attribute 'extract_profile'`

- [ ] **Step 3: 修改 `src/model/model_adapter.py`（两处：类内插入 extract_profile + 文件末尾追加模块级工具）**

**(a) 模块级工具追加在文件末尾（以下代码块 1）：**

```python
PROFILE_FIELDS = ["major", "grade", "skills", "interests", "career_goals", "output_preference"]
_LIST_FIELDS = {"skills", "interests", "career_goals"}
_MAJOR_RE = re.compile(
    r"(人工智能|计算机|软件工程|数据科学|电子信息|通信工程|自动化|电气工程|机械|材料|"
    r"化学|生物|医学|药学|数学|统计|物理|法学|经济|金融|会计|管理|市场营销|中文|"
    r"外语|英语|日语|教育|心理|新闻|设计|美术|音乐|建筑|土木)"
)
_GRADE_RE = re.compile(r"(大一|大二|大三|大四|研一|研二|研三|博一|博二|博三|已毕业|毕业[0-9一二三四五]年)")
_SKILL_RE = re.compile(r"(?:会|熟练|掌握|精通|熟悉|了解)(?:使用|运用)?[:：]?([^，。；、！？\n]{1,20})")
_GOAL_RE = re.compile(r"(?:想做|想当|想成为|目标是|目标岗位是|希望从事|希望成为|打算做|打算从事)[:：]?([^，。；、！？\n]{1,20})")


def _coerce_profile(data):
    """补齐缺省字段；空字段自动加入 missing_fields（双重保险，绝不虚构）。"""
    profile = {}
    for key in PROFILE_FIELDS:
        value = data.get(key)
        if key in _LIST_FIELDS:
            profile[key] = value if isinstance(value, list) else []
        else:
            profile[key] = value if isinstance(value, str) else ""
    missing = set(data.get("missing_fields") or [])
    for key in PROFILE_FIELDS:
        is_empty = not profile[key]
        if is_empty:
            missing.add(key)
    profile["missing_fields"] = sorted(missing)
    return profile


def _rule_based_profile(raw_input):
    """规则提取兜底（手册 3.2）：提取不到的内容一律置空并写入 missing_fields，绝不虚构。"""
    text = raw_input or ""
    major_match = _MAJOR_RE.search(text)
    grade_match = _GRADE_RE.search(text)
    partial = {
        "major": major_match.group(1) if major_match else "",
        "grade": grade_match.group(1) if grade_match else "",
        "skills": list(dict.fromkeys(_SKILL_RE.findall(text)))[:10],
        "interests": [],
        "career_goals": _GOAL_RE.findall(text)[:5],
        "output_preference": "",
        "missing_fields": [],
    }
    return _coerce_profile(partial)
```

**(b) `extract_profile` 方法插入到 ModelAdapter 类内部：定位 Task 3 代码中 `generate_json` 方法的最后一行 `        raise last_error`，在该行之后、类定义结束之前插入以下方法（缩进 4 空格，代码块 2）：**

```python
    def extract_profile(self, raw_input):
        """从原始输入提取用户画像。永不抛异常；最差返回规则兜底画像。

        返回 {"profile": {...}, "errors": [...], "used_fallback": bool,
              "elapsed_ms": int, "usage": dict|None}
        """
        started = time.perf_counter()
        self.last_usage = None
        try:
            data = self.generate_json(PROFILE_SYSTEM_PROMPT, raw_input, self.profile_schema)
        except ModelError as exc:
            logger.warning("extract_profile 降级到规则提取 code=%s", exc.code)
            return {
                "profile": _rule_based_profile(raw_input),
                "errors": [exc.to_error_dict()],
                "used_fallback": True,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "usage": None,
            }
        return {
            "profile": _coerce_profile(data),
            "errors": [],
            "used_fallback": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "usage": self.last_usage,
        }
```

- [ ] **Step 4: 创建 `tests/fixtures/profile_samples.json`（3 个固定样例，供测试与演示共用）**

```json
[
  {
    "id": "sample_1_full",
    "input": "我是计算机科学与技术专业的大三学生，会Python和C++，对人工智能和数据分析很感兴趣，毕业后想做算法工程师，希望输出内容详细一些。"
  },
  {
    "id": "sample_2_partial",
    "input": "软件工程大二，目前只会Java，还没想好以后做什么。"
  },
  {
    "id": "sample_3_minimal",
    "input": "帮我规划一下。"
  }
]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_model_adapter.py -v`
Expected: PASS（12 passed）

- [ ] **Step 6: 全量回归**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS（30 passed）

- [ ] **Step 7: 提交**

```bash
git add src/model/model_adapter.py tests/test_model_adapter.py tests/fixtures/profile_samples.json
git commit -m "feat(model): extract_profile 画像提取与规则兜底（TDD）"
```

---

### Task 5: 演示脚本与真实 API 验证

**Files:**
- Create: `scripts/demo_profile.py`
- Create: `outputs/week1/`（运行时生成，gitignore）

- [ ] **Step 1: 创建 `scripts/demo_profile.py`**

```python
"""第一周真实验证：/models 确认 + 3 个样例画像提取。

用法（项目根目录）：
  .venv/Scripts/python.exe scripts/demo_profile.py
环境变量 DEEPSEEK_API_KEY 缺失时会自动读取本地 .env（仅脚本层便利，客户端本身只读环境变量）。
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not os.environ.get("DEEPSEEK_API_KEY"):
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from src.model.deepseek_client import DeepSeekClient  # noqa: E402
from src.model.model_adapter import ModelAdapter  # noqa: E402


def main():
    client = DeepSeekClient()
    models = client.list_models()
    print(f"[1/2] /models 可用模型列表：{models}")
    if client.model not in models:
        print(f"[1/2] 警告：{client.model} 不在 /models 列表中，仍继续尝试调用")
    else:
        print(f"[1/2] 确认 {client.model} 可用")

    adapter = ModelAdapter(client)
    samples = json.loads(
        (ROOT / "tests" / "fixtures" / "profile_samples.json").read_text(encoding="utf-8")
    )
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n=== 样例 {i}/{len(samples)}：{sample['id']} ===")
        print(f"输入：{sample['input']}")
        result = adapter.extract_profile(sample["input"])
        result["sample_id"] = sample["id"]
        result["input"] = sample["input"]
        results.append(result)
        print(f"used_fallback={result['used_fallback']} elapsed_ms={result['elapsed_ms']}")
        print(f"token用量={result['usage']}")
        print("画像：")
        print(json.dumps(result["profile"], ensure_ascii=False, indent=2))
        if result["errors"]:
            print("错误：")
            print(json.dumps(result["errors"], ensure_ascii=False, indent=2))

    out_dir = ROOT / "outputs" / "week1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "profile_baseline_results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if not r["used_fallback"])
    print(f"\n[2/2] 结果已保存：{out_file}")
    print(f"[2/2] 汇总：{ok}/{len(results)} 个样例由 API 成功提取（其余走规则兜底）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行真实验证**

Run: `.venv/Scripts/python.exe scripts/demo_profile.py`

Expected（真实网络结果）：
- `[1/2]` 输出模型列表，且确认 `deepseek-v4-flash` 可用（若列表中没有该模型，脚本会警告但继续，此时需回看警告并考虑换模型名）
- 样例1、2 期望 `used_fallback=False`（画像字段基本正确）；样例3 期望全部字段为空且 `missing_fields` 含全部 6 个字段（证明不虚构）
- 若出现 API 错误（402 余额不足等），errors 字段必须是结构化格式 `{"module": "model", "code": ..., "message": ..., "retryable": ...}` 而非裸异常——这本身就是兜底路径的正常表现
- `outputs/week1/profile_baseline_results.json` 生成且内容完整

- [ ] **Step 3: 核对验收标准并提交**

核对（对照手册成员B验收标准）：
1. 密钥只在 .env 环境变量 → `git ls-files | grep -E '\.env$|111'` 无输出
2. 3 个样例稳定返回 user_profile JSON → 上一步结果文件可查
3. 不得虚构 → 样例3 的 missing_fields 齐全、无臆造内容
4. 超时/限流/余额/JSON 错误有明确提示 → 单元测试 TestErrorMapping / TestExtractProfile 已覆盖

```bash
git add scripts/demo_profile.py
git commit -m "feat(model): 真实 API 验证脚本（/models + 3 样例画像基线）"
```

注意：`outputs/week1/` 已被 gitignore，验证证据保留在本地磁盘，交接时随目录拷贝给 A。

---

### Task 6: 接口说明与收尾自查

**Files:**
- Create: `src/model/README.md`
- Modify: 无

- [ ] **Step 1: 创建 `src/model/README.md`（48h 清单"提供稳定函数和失败返回格式"的文档化）**

````markdown
# src/model —— 模型模块（成员B）

DeepSeek 调用、Prompt、画像提取、规划生成与反馈调整。本周交付：客户端 + 适配器 + 画像基线。

## 依赖与环境

- 环境变量：`DEEPSEEK_API_KEY`（必填，密钥绝不写入代码/日志/仓库）
- 依赖：`requests`、`jsonschema`；测试：`pytest`

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
````

- [ ] **Step 2: 全量测试 + 目录结构核对**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: PASS（30 passed）

Run: `git status --short`
Expected: 除 `src/model/README.md` 外无未跟踪文件（`111.txt`、`.env`、`outputs/` 应显示为被忽略，不出现在 git status 中）

- [ ] **Step 3: 最终提交**

```bash
git add src/model/README.md
git commit -m "docs(model): 模型模块接口说明与失败格式文档"
```

- [ ] **Step 4: 交接给 A 的信息（口头/消息，无需提交）**

把以下内容发给 A：`extract_profile` 接口签名与返回结构（见 README）、3 个样例的验证结果文件路径（`outputs/week1/profile_baseline_results.json`）、错误条目格式约定（错误对象直接可写入 `SessionState.errors`）。

---

## 验收对照（全部任务完成后自查）

| 手册成员B标准 | 证据 |
|--------------|------|
| 密钥只在 DEEPSEEK_API_KEY 环境变量 | `git ls-files` 无密钥文件；客户端只读环境变量 |
| 画像 JSON 解析稳定（本周基线 3 样例） | `outputs/week1/profile_baseline_results.json` |
| 不得虚构、缺失写 missing_fields | Prompt 硬约束 + `_coerce_profile` + `test_fallback_never_fabricates` |
| 超时/限流/余额/JSON 错误明确提示和日志 | 类型化错误 + to_error_dict + logger("model") |
| 相关成员能够调用 | `src/model/README.md` + demo 脚本 |
