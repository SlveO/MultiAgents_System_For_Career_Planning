from __future__ import annotations

import json
from typing import Dict, Generator, Iterable, Optional, Tuple

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
        }

    def _url(self) -> str:
        # DeepSeek OpenAI-compatible endpoint
        return f"{self.base_url}/chat/completions"

    def plan(self, prompt: str, model: Optional[str] = None) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self._url(),
                headers=self._headers(),
                json=self._payload(prompt, model=model, stream=False),
            )
            r.raise_for_status()
            obj = r.json()
        return obj["choices"][0]["message"]["content"]

    def plan_stream(self, prompt: str, model: Optional[str] = None) -> Generator[str, None, None]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                self._url(),
                headers=self._headers(),
                json=self._payload(prompt, model=model, stream=True),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content") or delta.get("reasoning_content") or ""
                    if token:
                        yield token


