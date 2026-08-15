from __future__ import annotations

import re
from typing import Any


_DROP_KEYS = {
    "raw_output",
    "quote",
    "document_paths",
    "image_paths",
    "audio_paths",
    "video_paths",
}


def redact_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)([A-Z]:\\Users\\)[^\\\s]+",
        r"\1[REDACTED_USER]",
        text,
    )
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[REDACTED_EMAIL]",
        text,
    )
    text = re.sub(
        r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)",
        "[REDACTED_PHONE]",
        text,
    )
    text = re.sub(
        r"(?<!\d)\d{17}[0-9Xx](?!\d)",
        "[REDACTED_ID_CARD]",
        text,
    )
    text = re.sub(
        r"(身份证号?|身份证)\s*[:：]?\s*[0-9Xx-]{8,20}",
        r"\1：[REDACTED_ID_CARD]",
        text,
    )
    text = re.sub(
        r"(学号|student\s*id)\s*[:：]?\s*[A-Za-z0-9-]{6,20}",
        r"\1：[REDACTED_STUDENT_ID]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(姓名|名字)\s*[:：]?\s*[\u4e00-\u9fff]{2,4}",
        r"\1：[REDACTED_NAME]",
        text,
    )
    text = re.sub(
        r"我叫\s*[\u4e00-\u9fff]{2,4}",
        "我叫[REDACTED_NAME]",
        text,
    )
    return text


def redact_data(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_data(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): redact_data(item)
            for key, item in value.items()
            if str(key) not in _DROP_KEYS
        }
    return value
