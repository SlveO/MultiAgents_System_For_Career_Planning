from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback, _safe_confidence


class VideoPerceptionAgent:
    def __init__(
        self,
        image_model_path: str = "./models/Qwen3-VL-2B-Instruct",
        max_frames: int = 5,
        sample_interval: int = 5,
    ) -> None:
        self.image_model_path = image_model_path
        self.max_frames = max_frames
        self.sample_interval = sample_interval
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            try:
                from project.agents.image import ImageProcessor
            except Exception:
                from agents.image import ImageProcessor
            self._processor = ImageProcessor(model_path=self.image_model_path)
        return self._processor

    def _extract_frames(self, video_path: str) -> List[str]:
        try:
            import cv2
        except Exception:
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0
        duration = total_frames / fps if fps > 0 else 0

        frame_paths: List[str] = []
        positions = list(range(0, int(duration), self.sample_interval))
        if len(positions) > self.max_frames:
            step = max(1, len(positions) // self.max_frames)
            positions = positions[::step][:self.max_frames]

        tmp_dir = tempfile.mkdtemp(prefix="video_frames_")
        for sec in positions:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_path = os.path.join(tmp_dir, f"frame_{sec}s.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            if len(frame_paths) >= self.max_frames:
                break

        cap.release()
        return frame_paths

    def perceive(self, video_path: str) -> PerceptionResult:
        p = Path(video_path)
        if not p.exists():
            return PerceptionResult(
                modality="video",
                summary=f"Video not found: {video_path}",
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=["Video file path is incorrect"],
                raw_output="",
            )

        size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        frame_paths = self._extract_frames(video_path)

        if not frame_paths:
            return PerceptionResult(
                modality="video",
                summary=f"Video detected: {p.name} ({size_mb}MB), no frames could be extracted",
                facts=[f"Video file: {p.name}", f"Size: {size_mb}MB"],
                evidence=[],
                confidence=_safe_confidence(0.4),
                missing_info=["Frame extraction failed, cv2 may not be available"],
                raw_output=f"[video:{p.name}] size={size_mb}MB",
            )

        processor = self._get_processor()
        descriptions: List[str] = []
        for fp in frame_paths:
            try:
                desc = processor.analyze(fp, question="Describe what is shown in this video frame.")
                descriptions.append(f"[{Path(fp).name}] {desc}")
            except Exception as exc:
                descriptions.append(f"[{Path(fp).name}] description failed: {exc}")

        # Clean up temp frames
        tmp_dir = os.path.dirname(frame_paths[0]) if frame_paths else ""
        for fp in frame_paths:
            try:
                os.remove(fp)
            except Exception:
                pass
        if tmp_dir:
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass

        combined = "\n".join(descriptions)
        facts = _extract_facts_fallback(combined, limit=8)
        processor.unload()

        return PerceptionResult(
            modality="video",
            summary=f"Analyzed {p.name} ({size_mb}MB, {len(frame_paths)} frames)",
            facts=facts,
            evidence=[EvidenceItem(source=video_path, quote=combined[:500])],
            confidence=_safe_confidence(0.6 if facts else 0.4),
            missing_info=[],
            raw_output=combined,
        )
