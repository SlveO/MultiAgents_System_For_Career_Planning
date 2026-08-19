from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterable, Optional

import httpx

try:
    from .settings import get_settings
except ImportError:
    from project.core.settings import get_settings


class BrainClient:
    def plan(self, prompt: str, model: Optional[str] = None) -> str:
        raise NotImplementedError

    def plan_stream(self, prompt: str, model: Optional[str] = None) -> Iterable[str]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class BrainClientError(RuntimeError):
    code = "BRAIN_ERROR"
    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    def to_error_dict(self) -> Dict[str, Any]:
        return {
            "module": "brain",
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }


class BrainConfigError(BrainClientError):
    code = "BRAIN_CONFIG"


class BrainAuthError(BrainClientError):
    code = "BRAIN_AUTH"


class BrainBalanceError(BrainClientError):
    code = "BRAIN_BALANCE"


class BrainRateLimitError(BrainClientError):
    code = "BRAIN_RATE_LIMIT"
    retryable = True


class BrainTimeoutError(BrainClientError):
    code = "BRAIN_TIMEOUT"
    retryable = True


class BrainServerError(BrainClientError):
    code = "BRAIN_SERVER"
    retryable = True


class BrainHTTPError(BrainClientError):
    code = "BRAIN_HTTP"


class BrainResponseError(BrainClientError):
    code = "BRAIN_RESPONSE"
    retryable = True


_STATUS_TO_ERROR = {
    401: BrainAuthError,
    402: BrainBalanceError,
    429: BrainRateLimitError,
}

_STATUS_MESSAGES = {
    401: "DeepSeek API 密钥无效",
    402: "DeepSeek API 账户余额不足",
    429: "DeepSeek API 请求受到限流",
}


class DeepSeekBrainClient(BrainClient):
    def __init__(self):
        s = get_settings()
        self.api_key = s.deepseek_api_key
        self.base_url = s.deepseek_base_url.rstrip("/")
        self.default_model = s.brain_default_model
        self.timeout = s.brain_timeout_seconds

    @property
    def model_name(self) -> str:
        return self.default_model

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, prompt: str, model: Optional[str], stream: bool) -> Dict:
        return {
            "model": model or self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "stream": stream,
            "thinking": {"type": "disabled"},
        }

    def _url(self) -> str:
        # DeepSeek OpenAI-compatible endpoint
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return

        error_class = _STATUS_TO_ERROR.get(response.status_code)
        if error_class is None:
            error_class = BrainServerError if response.status_code >= 500 else BrainHTTPError
        message = _STATUS_MESSAGES.get(
            response.status_code,
            f"DeepSeek API 返回 HTTP {response.status_code}",
        )
        raise error_class(message, status_code=response.status_code)

    @staticmethod
    def _extract_content(obj: Any) -> str:
        try:
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BrainResponseError("DeepSeek API 响应缺少规划文本") from exc
        if not isinstance(content, str) or not content.strip():
            raise BrainResponseError("DeepSeek API 返回了空规划文本")
        return content

    def plan(self, prompt: str, model: Optional[str] = None) -> str:
        if not self.api_key:
            raise BrainConfigError("DEEPSEEK_API_KEY 未设置")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._url(),
                    headers=self._headers(),
                    json=self._payload(prompt, model=model, stream=False),
                )
                self._raise_for_status(response)
                try:
                    obj = response.json()
                except ValueError as exc:
                    raise BrainResponseError("DeepSeek API 返回了无效 JSON") from exc
        except httpx.TimeoutException as exc:
            raise BrainTimeoutError("DeepSeek API 请求超时") from exc
        except httpx.RequestError as exc:
            raise BrainHTTPError("无法连接 DeepSeek API") from exc
        return self._extract_content(obj)

    def plan_stream(self, prompt: str, model: Optional[str] = None) -> Generator[str, None, None]:
        if not self.api_key:
            raise BrainConfigError("DEEPSEEK_API_KEY 未设置")

        yielded_content = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    self._url(),
                    headers=self._headers(),
                    json=self._payload(prompt, model=model, stream=True),
                ) as response:
                    self._raise_for_status(response)
                    for line in response.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        choices = obj.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token = delta.get("content") or ""
                        if token:
                            yielded_content = True
                            yield token
        except httpx.TimeoutException as exc:
            raise BrainTimeoutError("DeepSeek API 流式请求超时") from exc
        except httpx.RequestError as exc:
            raise BrainHTTPError("无法连接 DeepSeek API") from exc
        if not yielded_content:
            raise BrainResponseError("DeepSeek API 流式响应没有规划文本")


