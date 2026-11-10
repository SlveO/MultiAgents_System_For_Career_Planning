from __future__ import annotations

from typing import Callable


FEEDBACK_OPTIONS = ("过短", "合适", "过于详细")
_NUMBERED_OPTIONS = {"1": "过短", "2": "合适", "3": "过于详细"}


def collect_feedback(input_fn: Callable[[str], str] = input, *, allow_finish: bool = False) -> str | None:
    prompt = "请选择规划详细程度：1=过短，2=合适，3=过于详细"
    if allow_finish:
        prompt += "，输入‘结束’保留当前版本"
    prompt += "\n> "
    while True:
        value = (input_fn(prompt) or "").strip()
        if allow_finish and value in {"结束", "退出", "结束反馈"}:
            return None
        normalized = _NUMBERED_OPTIONS.get(value, value)
        if normalized in FEEDBACK_OPTIONS:
            return normalized
