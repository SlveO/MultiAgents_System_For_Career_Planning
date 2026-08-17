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
