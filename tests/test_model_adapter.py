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
