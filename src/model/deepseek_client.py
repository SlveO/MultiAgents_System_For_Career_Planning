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
