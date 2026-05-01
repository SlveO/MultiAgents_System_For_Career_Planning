from __future__ import annotations

import re
from typing import List

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback, _safe_confidence

SKILL_KEYWORDS = [
    "python", "java", "c\\+\\+", "sql", "机器学习", "深度学习", "数据分析",
    "数据挖掘", "爬虫", "tensorflow", "pytorch", "nlp", "计算机视觉",
    "项目管理", "scrum", "沟通", "演讲", "写作", "英语",
    "excel", "tableau", "power\\s*bi", "spark", "hadoop",
    "前端", "后端", "全栈", "运维", "测试", "产品", "设计",
]

STRENGTH_PATTERNS = re.compile(
    r"(?:熟练|精通|掌握|擅长|熟悉|有.*经验|做过|负责过|参与过)"
    r"(?:[一-鿿\w]+(?:、[一-鿿\w]+)*)",
)

WEAKNESS_PATTERNS = re.compile(
    r"(?:缺乏|不足|短板|薄弱|不太会|不熟|不会|没有.*经验)"
    r"(?:[一-鿿\w]+(?:、[一-鿿\w]+)*)",
)

INTEREST_PATTERNS = re.compile(
    r"(?:想做|想转|感兴趣|喜欢|希望从事|目标是|方向是|岗位是)"
    r"(?:[一-鿿\w]+(?:、[一-鿿\w]+)*)",
)


class TextPerceptionAgent:
    """Rule-based text perception — no model dependency. Extracts structured facts via regex/keyword analysis."""

    def __init__(self):
        pass

    def perceive(self, user_text: str) -> PerceptionResult:
        text = user_text or ""

        facts = self._extract_facts(text)
        strengths = self._extract_strengths(text)
        weaknesses = self._extract_weaknesses(text)
        interests = self._extract_interests(text)
        skills = self._match_skills(text)

        all_facts = facts + [
            f"技能关键词: {', '.join(skills)}" if skills else "",
            f"优势: {', '.join(strengths)}" if strengths else "",
            f"短板: {', '.join(weaknesses)}" if weaknesses else "",
            f"意向: {', '.join(interests)}" if interests else "",
        ]
        all_facts = [f for f in all_facts if f.strip()]

        evidence_quotes = self._extract_evidence(text, strengths + weaknesses + interests)

        missing: List[str] = []
        if not strengths:
            missing.append("未明确描述个人优势或技能")
        if not interests:
            missing.append("未明确描述目标岗位或转行方向")
        if "时间" not in text and "月" not in text and "周" not in text:
            missing.append("未提供时间预算")

        confidence = 0.55
        if skills and strengths:
            confidence = 0.70
        if skills and strengths and interests:
            confidence = 0.80

        return PerceptionResult(
            modality="text",
            summary=self._build_summary(skills, strengths, weaknesses, interests),
            facts=all_facts[:10],
            evidence=evidence_quotes[:4],
            confidence=_safe_confidence(confidence),
            missing_info=missing or ["建议补充目标岗位、当前能力水平、时间预算"],
            raw_output="",
        )

    @staticmethod
    def _build_summary(skills: List[str], strengths: List[str], weaknesses: List[str], interests: List[str]) -> str:
        parts = []
        if skills:
            parts.append(f"检测到技能: {', '.join(skills[:6])}")
        if strengths:
            parts.append(f"优势领域: {', '.join(strengths[:4])}")
        if weaknesses:
            parts.append(f"待提升: {', '.join(weaknesses[:4])}")
        if interests:
            parts.append(f"意向方向: {', '.join(interests[:3])}")
        return "；".join(parts) if parts else "已解析用户文本输入（规则模式）"

    @staticmethod
    def _extract_facts(text: str) -> List[str]:
        return _extract_facts_fallback(text, limit=5)

    @staticmethod
    def _extract_strengths(text: str) -> List[str]:
        matches = STRENGTH_PATTERNS.findall(text)
        return list(dict.fromkeys(m.strip() for m in matches if m.strip()))[:5]

    @staticmethod
    def _extract_weaknesses(text: str) -> List[str]:
        matches = WEAKNESS_PATTERNS.findall(text)
        return list(dict.fromkeys(m.strip() for m in matches if m.strip()))[:5]

    @staticmethod
    def _extract_interests(text: str) -> List[str]:
        matches = INTEREST_PATTERNS.findall(text)
        return list(dict.fromkeys(m.strip() for m in matches if m.strip()))[:5]

    @classmethod
    def _match_skills(cls, text: str) -> List[str]:
        lower = text.lower()
        found = []
        for kw in SKILL_KEYWORDS:
            if re.search(kw, lower):
                found.append(kw.replace("\\", ""))
        return found[:8]

    @staticmethod
    def _extract_evidence(text: str, items: List[str], max_len: int = 200) -> List[EvidenceItem]:
        result = []
        for item in items[:4]:
            idx = text.find(item)
            if idx >= 0:
                start = max(0, idx - 10)
                end = min(len(text), idx + len(item) + 40)
                quote = text[start:end].strip()
            else:
                quote = item
            result.append(EvidenceItem(source="text_input", quote=quote[:max_len]))
        if not result:
            result.append(EvidenceItem(source="text_input", quote=text[:200]))
        return result
