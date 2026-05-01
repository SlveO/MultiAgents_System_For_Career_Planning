from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback, _safe_confidence


def _load_image_processor():
    try:
        from project.agents.image import ImageProcessor
    except Exception:
        from agents.image import ImageProcessor
    return ImageProcessor


class ImagePerceptionAgent:
    def __init__(self, model_path: str):
        ImageProcessor = _load_image_processor()
        self.processor = ImageProcessor(model_path=model_path)

    def perceive(self, image_path: str, user_goal: str, user_text: str = '') -> PerceptionResult:
        ask = '请基于图像提取与职业规划相关的信息，输出可验证事实。'
        if user_goal:
            ask += f'\n用户目标：{user_goal}'
        if user_text:
            ask += f'\n用户补充：{user_text[:500]}'

        raw = self.processor.analyze(image_path, question=ask)
        facts = _extract_facts_fallback(raw)
        return PerceptionResult(
            modality='image',
            summary=f'已解析图像 {Path(image_path).name}',
            facts=facts,
            evidence=[EvidenceItem(source=image_path, quote=raw[:250])],
            confidence=_safe_confidence(0.65 if facts else 0.45),
            missing_info=[] if facts else ['图像信息不足，建议补充文本上下文'],
            raw_output=raw,
        )

    def unload(self) -> None:
        self.processor.unload()
