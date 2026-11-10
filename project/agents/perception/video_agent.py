from __future__ import annotations

import math
import tempfile
from pathlib import Path

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
        if not 1 <= max_frames <= 64 or sample_interval < 1:
            raise ValueError("Video sampling requires 1-64 frames and a positive interval")
        self.image_model_path = image_model_path
        self.max_frames = max_frames
        self.sample_interval = sample_interval
        self._processor = None
        self._frame_directory = None

    def _get_processor(self):
        if self._processor is None:
            try:
                from project.agents.image import ImageProcessor
            except ImportError:
                from agents.image import ImageProcessor
            self._processor = ImageProcessor(model_path=self.image_model_path)
        return self._processor

    def _extract_frames(self, video_path: str) -> list[str]:
        try:
            import cv2
        except ImportError:
            return []

        capture = cv2.VideoCapture(str(video_path))
        temporary = None
        frame_paths = []
        try:
            if not capture.isOpened():
                return []
            fps = capture.get(cv2.CAP_PROP_FPS)
            total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            duration = total_frames / fps if math.isfinite(total_frames) and total_frames > 0 else 0
            candidates = max(1, int(max(0, duration - 1)) // self.sample_interval + 1)
            count = min(candidates, self.max_frames)
            # Compute only the selected timestamps, even for a very long video.
            positions = [0] if count == 1 else [
                (index * (candidates - 1) // (count - 1)) * self.sample_interval
                for index in range(count)
            ]
            temporary = tempfile.TemporaryDirectory(prefix="career-video-")
            self._frame_directory = temporary
            for second in positions:
                capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                success, frame = capture.read()
                if not success:
                    continue
                path = Path(temporary.name) / f"frame_{second}s.jpg"
                if cv2.imwrite(str(path), frame):
                    frame_paths.append(str(path))
            return frame_paths
        except Exception:
            frame_paths.clear()
            raise
        finally:
            capture.release()
            if temporary is not None and not frame_paths:
                # OpenCV may write a partial image and still report failure.
                # Remove the entire directory that this call created, including
                # artifacts absent from frame_paths.
                temporary.cleanup()
                self._frame_directory = None

    def perceive(self, video_path: str) -> PerceptionResult:
        path = Path(video_path)
        if not path.is_file():
            return PerceptionResult(
                modality="video", summary=f"Video not found: {path.name}",
                confidence=0.0, missing_info=["Video file path is incorrect"],
            )
        frame_paths = []
        descriptions, evidence, warnings = [], [], []
        processor = None
        try:
            frame_paths = self._extract_frames(video_path)
            if not frame_paths:
                warnings.append("No video frames were extracted; check the video and optional OpenCV dependency")
            else:
                processor = self._get_processor()
                for frame_path in frame_paths:
                    frame_name = Path(frame_path).stem
                    try:
                        description = processor.analyze(
                            frame_path,
                            question="提取此视频帧中可见的职业规划相关事实；不要补充或推测未出现的信息。",
                        ).strip()
                        if description:
                            descriptions.append(f"[{frame_name}] {description}")
                            evidence.append(EvidenceItem(
                                source=f"{video_path}#frame={frame_name}", quote=description[:500],
                            ))
                        else:
                            warnings.append(f"Video frame {frame_name} returned no usable content")
                    except Exception as exc:
                        warnings.append(f"Video frame {frame_name} perception failed ({type(exc).__name__})")
        except Exception as exc:
            warnings.append(f"Video perception failed ({type(exc).__name__})")
        finally:
            if self._frame_directory is not None:
                self._frame_directory.cleanup()
                self._frame_directory = None
            for frame_path in frame_paths:
                Path(frame_path).unlink(missing_ok=True)
            for directory in {Path(item).parent for item in frame_paths}:
                try:
                    directory.rmdir()
                except OSError:
                    pass
            if processor is not None:
                try:
                    processor.unload()
                except Exception as exc:
                    warnings.append(f"Video processor cleanup failed ({type(exc).__name__})")

        combined = "\n".join(descriptions)
        facts = _extract_facts_fallback(combined, limit=8)
        # Frame sampling does not analyse sound or every event in the video.
        if facts:
            warnings.append(f"Video sampled {len(frame_paths)} frames; audio and unsampled intervals were not analysed")
        return PerceptionResult(
            modality="video", summary=f"Analyzed {path.name} ({len(descriptions)}/{len(frame_paths)} frames)",
            facts=facts, evidence=evidence,
            confidence=_safe_confidence(0.6) if facts else 0.0,
            missing_info=warnings, raw_output=combined,
        )
