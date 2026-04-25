from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback


class AudioPerceptionAgent:
    def __init__(self):
        self.asr_pipeline = None

    def _lazy_load(self):
        if self.asr_pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
            self.asr_pipeline = pipeline(
                'automatic-speech-recognition',
                model='openai/whisper-small',
                device='cuda:0',
            )
        except Exception:
            self.asr_pipeline = False

    def perceive(self, audio_path: str) -> PerceptionResult:
        p = Path(audio_path)
        if not p.exists():
            return PerceptionResult(
                modality='audio',
                summary=f'音频不存在: {audio_path}',
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=['请检查音频路径'],
                raw_output='',
            )

        self._lazy_load()
        if self.asr_pipeline is False:
            return PerceptionResult(
                modality='audio',
                summary='ASR 不可用',
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=['当前环境不可用 whisper，建议先安装模型依赖'],
                raw_output='',
            )

        result = self.asr_pipeline(str(p))
        text = result['text'] if isinstance(result, dict) else str(result)
        facts = _extract_facts_fallback(text, limit=6)
        return PerceptionResult(
            modality='audio',
            summary=f'已完成音频转写 {p.name}',
            facts=facts,
            evidence=[EvidenceItem(source=audio_path, quote=text[:260])],
            confidence=0.6,
            missing_info=[],
            raw_output=text,
        )
