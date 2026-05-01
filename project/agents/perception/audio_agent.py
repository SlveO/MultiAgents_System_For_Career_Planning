from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback, _safe_confidence


class AudioPerceptionAgent:
    def __init__(
        self,
        model_path: str = "./models/openai-mirror/whisper-small",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.asr_pipeline = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _lazy_load(self):
        if self.asr_pipeline is not None:
            return
        try:
            from transformers import pipeline
            device = self._resolve_device()
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_path,
                device=device if device != "cpu" else -1,
            )
        except Exception:
            try:
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_path,
                    device=-1,
                )
            except Exception:
                self.asr_pipeline = False

    def perceive(self, audio_path: str) -> PerceptionResult:
        p = Path(audio_path)
        if not p.exists():
            return PerceptionResult(
                modality="audio",
                summary=f"Audio not found: {audio_path}",
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=["Audio file path is incorrect"],
                raw_output="",
            )

        self._lazy_load()
        if self.asr_pipeline is False:
            return PerceptionResult(
                modality="audio",
                summary="ASR unavailable",
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=["Whisper model not available, please install model dependencies"],
                raw_output="",
            )

        result = self.asr_pipeline(str(p))
        text = result["text"] if isinstance(result, dict) else str(result)
        facts = _extract_facts_fallback(text, limit=6)
        return PerceptionResult(
            modality="audio",
            summary=f"Transcribed {p.name}",
            facts=facts,
            evidence=[EvidenceItem(source=audio_path, quote=text[:260])],
            confidence=_safe_confidence(0.6),
            missing_info=[],
            raw_output=text,
        )
