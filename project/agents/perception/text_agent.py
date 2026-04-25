from __future__ import annotations

from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_json_object, _extract_facts_fallback, _safe_confidence


def _load_text_processor():
    try:
        from project.agents.text import TextProcessor
    except Exception:
        from agents.text import TextProcessor
    return TextProcessor


class TextPerceptionAgent:
    def __init__(self, model_path: str):
        TextProcessor = _load_text_processor()
        self.processor = TextProcessor(model_path=model_path)

    def perceive(self, user_text: str) -> PerceptionResult:
        prompt = f"""
请从以下用户输入中抽取结构化信息，并输出严格 JSON：
{
  "summary": "一句话总结",
  "facts": ["事实1", "事实2"],
  "evidence": ["引用原文片段1", "引用原文片段2"],
  "confidence": 0.0,
  "missing_info": ["缺失信息1"]
}

用户输入：
{user_text}
"""
        raw = self.processor.generate(prompt, stream=False)['response']
        obj = _extract_json_object(raw)

        if obj:
            evidence = [
                EvidenceItem(source='text_input', quote=s)
                for s in obj.get('evidence', [])[:4]
                if isinstance(s, str)
            ]
            return PerceptionResult(
                modality='text',
                summary=str(obj.get('summary', '')).strip() or '已解析用户文本输入',
                facts=[str(x) for x in obj.get('facts', []) if str(x).strip()],
                evidence=evidence,
                confidence=_safe_confidence(obj.get('confidence', 0.6)),
                missing_info=[str(x) for x in obj.get('missing_info', []) if str(x).strip()],
                raw_output=raw,
            )

        return PerceptionResult(
            modality='text',
            summary='文本解析完成（回退模式）',
            facts=_extract_facts_fallback(user_text),
            evidence=[EvidenceItem(source='text_input', quote=user_text[:200])],
            confidence=0.45,
            missing_info=['建议补充目标岗位、时间预算、当前能力水平'],
            raw_output=raw,
        )
