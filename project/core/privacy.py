from __future__ import annotations

import re
import hashlib
from typing import Any


_DROP_KEYS = {
    "source",
    "raw_output",
    "quote",
    "document_paths",
    "image_paths",
    "audio_paths",
    "video_paths",
}

_HEX_20 = re.compile(r"[0-9a-fA-F]{20}\Z")
_HEX_32 = re.compile(r"[0-9a-fA-F]{32}\Z")
_HEX_64 = re.compile(r"[0-9a-fA-F]{64}\Z")
_GRAPH_ID = re.compile(r"[npe]_[0-9a-fA-F]{20}\Z")
_SESSION_KEY = re.compile(r"session-[0-9a-fA-F]{64}\Z")
_KNOWLEDGE_VERSION = re.compile(r"(?:kb|seed)-[0-9a-fA-F]{20}(?:-[0-9a-fA-F]{8})?\Z")
_MACHINE_FIELDS = {
    "plan_id": (_HEX_32,),
    "request_id": (_HEX_32, _HEX_64),
    "facts_sha256": (_HEX_64,),
    "source_text_sha256": (_HEX_64,),
    "content_sha256": (_HEX_64,),
    "version_id": (_KNOWLEDGE_VERSION,),
    "source_id": (_HEX_20, _HEX_32, _HEX_64),
    "evidence_id": (_HEX_20,),
    "question_id": (_HEX_20,),
    "id": (_GRAPH_ID,),
    "from": (_GRAPH_ID,),
    "to": (_GRAPH_ID,),
    "session_id": (_SESSION_KEY,),
}
_MACHINE_LIST_FIELDS = {
    "evidence_ids": _HEX_20,
    "resolved_evidence_ids": _HEX_20,
    "asked_keys": _HEX_20,
    "provenance_ids": _GRAPH_ID,
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
            str(key): _redact_field(str(key), item)
            for key, item in value.items()
            if str(key) not in _DROP_KEYS
        }
    return value


def _redact_field(key: str, value: Any) -> Any:
    """Keep typed machine identities intact; never exempt arbitrary field text.

    Hexadecimal digests can contain a digit run resembling a telephone number.
    Replacing that run would break hashes, foreign keys and durable replay. Only
    known field names paired with their precise machine formats are exempted.
    """
    patterns = _MACHINE_FIELDS.get(key, ())
    if isinstance(value, str) and any(pattern.fullmatch(value) for pattern in patterns):
        return value
    pattern = _MACHINE_LIST_FIELDS.get(key)
    if pattern is not None and isinstance(value, (list, tuple)):
        return [item if isinstance(item, str) and pattern.fullmatch(item) else redact_data(item)
                for item in value]
    return redact_data(value)


def session_key(value: str) -> str:
    """Keep ordinary session labels; map identifiers with PII to a stable key."""
    if _SESSION_KEY.fullmatch(value):
        return value
    if redact_text(value) != value:
        return "session-" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    return value
