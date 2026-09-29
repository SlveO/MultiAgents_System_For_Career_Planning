from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback, _safe_confidence
from project.core.gpu import require_cuda
from project.core.settings import get_settings


class AudioPerceptionAgent:
    def __init__(
        self,
        model_path: str = "./models/openai-mirror/whisper-small",
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device or get_settings().local_model_device
        self.asr_pipeline = None

    def _resolve_device(self) -> str:
        return require_cuda(self.device)

    def _lazy_load(self):
        if self.asr_pipeline is not None:
            return
        device = self._resolve_device()
        if not Path(self.model_path).is_dir():
            raise RuntimeError("Audio model must already exist locally; automatic download is disabled")
        from transformers import pipeline
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition", model=self.model_path, device=device,
            model_kwargs={"local_files_only": True},
        )

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
