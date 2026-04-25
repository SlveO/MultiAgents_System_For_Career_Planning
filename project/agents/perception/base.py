from __future__ import annotations

import json
import re
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult


def _extract_json_object(text: str) -> Optional[dict]:
    text = (text or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    a = text.find('{')
    b = text.rfind('}')
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b+1])
        except Exception:
            return None
    return None


def _extract_facts_fallback(text: str, limit: int = 5) -> List[str]:
    lines = [x.strip('-* \t') for x in re.split(r'[\n。；;]', text or '') if x.strip()]
    return lines[:limit]


def _safe_confidence(value, default=0.6, minimum=0.35, maximum=0.95) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(minimum, min(maximum, v))
