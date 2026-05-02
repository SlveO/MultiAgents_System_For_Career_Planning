from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RoutedInput:
    mode: str
    text_content: str
    image_paths: List[str]
    file_paths: List[str]
    audio_video_paths: List[str]
    raw_input: str


class InputClassifier:
    """
    Unified input classifier.

    Supported modes:
    - text
    - image
    - file (documents: Word/PDF/Excel and related office/text files)
    - audio_video
    - multimodal (text + image)
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
    }
    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".tsv",
        ".ods",
        ".txt",
        ".md",
    }
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

    PATH_PATTERNS: Tuple[re.Pattern[str], ...] = (
        re.compile(r"^[A-Za-z]:[\\/].+\.[A-Za-z0-9]{2,6}$"),  # Windows absolute
        re.compile(r"^[/~].+\.[A-Za-z0-9]{2,6}$"),  # Unix absolute
        re.compile(r"^\.\.?[\\/].+\.[A-Za-z0-9]{2,6}$"),  # relative with ./ or ../
        re.compile(r"^[^<>:\"|?*\n\r\t]+\.[A-Za-z0-9]{2,6}$"),  # simple filename
    )

    TOKEN_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'|(\S+)')

    def classify(self, input_str: str) -> Dict[str, Any]:
        routed = self._classify_to_dataclass(input_str)
        image_details = [self._build_path_info(p) for p in routed.image_paths]
        file_details = [self._build_path_info(p) for p in routed.file_paths]
        audio_video_details = [self._build_path_info(p) for p in routed.audio_video_paths]
        return {
            "mode": routed.mode,
            "text_content": routed.text_content,
            "image_paths": routed.image_paths,
            "file_paths": routed.file_paths,
            "audio_video_paths": routed.audio_video_paths,
            "image_details": image_details,
            "file_details": file_details,
            "audio_video_details": audio_video_details,
            "raw_input": routed.raw_input,
        }

    def validate(self, classification: Dict[str, Any]) -> Tuple[bool, str]:
        mode = classification.get("mode", "")

        if mode == "text":
            return True, "text input"

        if mode == "multimodal":
            if not classification.get("text_content", "").strip():
                return False, "multimodal input requires text context"
            if not self._has_existing_path(classification.get("image_details", [])):
                return False, "no valid image file found"
            return True, "multimodal input"

        if mode == "image":
            if not self._has_existing_path(classification.get("image_details", [])):
                return False, "no valid image file found"
            return True, "image input"

        if mode == "file":
            if not self._has_existing_path(classification.get("file_details", [])):
                return False, "no valid document file found"
            return True, "document input"

        if mode == "audio_video":
            if not self._has_existing_path(classification.get("audio_video_details", [])):
                return False, "no valid audio/video file found"
            return True, "audio/video input"

        return False, f"unsupported mode: {mode}"

    def _classify_to_dataclass(self, input_str: str) -> RoutedInput:
        raw = (input_str or "").strip()
        if not raw:
            return RoutedInput(
                mode="text",
                text_content="",
                image_paths=[],
                file_paths=[],
                audio_video_paths=[],
                raw_input=raw,
            )

        candidates = self._extract_file_candidates(raw)
        categories: Dict[str, List[str]] = {
            "image": [],
            "file": [],
            "audio_video": [],
        }
        matched_candidates: List[str] = []
        for candidate in candidates:
            category = self._categorize_path(candidate)
            if category:
                categories[category].append(candidate)
                matched_candidates.append(candidate)

        text_content = self._remove_tokens(raw, matched_candidates)

        image_paths = self._dedupe(categories["image"])
        file_paths = self._dedupe(categories["file"])
        audio_video_paths = self._dedupe(categories["audio_video"])

        category_count = sum(bool(v) for v in (image_paths, file_paths, audio_video_paths))
        if category_count == 0:
            mode = "text"
        elif category_count > 1:
            mode = "file"
        elif image_paths and text_content:
            mode = "multimodal"
        elif image_paths:
            mode = "image"
        elif file_paths:
            mode = "file"
        else:
            mode = "audio_video"

        return RoutedInput(
            mode=mode,
            text_content=text_content,
            image_paths=image_paths,
            file_paths=file_paths,
            audio_video_paths=audio_video_paths,
            raw_input=raw,
        )

    def _extract_file_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        for m in self.TOKEN_RE.finditer(text):
            token = m.group(1) or m.group(2) or m.group(3) or ""
            token = token.strip().strip(",;")
            if not token:
                continue
            if self._looks_like_path(token):
                candidates.append(token)
        return candidates

    def _looks_like_path(self, value: str) -> bool:
        if not value or "." not in value:
            return False
        return any(pattern.search(value) for pattern in self.PATH_PATTERNS)

    def _categorize_path(self, path_str: str) -> Optional[str]:
        ext = Path(path_str).suffix.lower()
        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        if ext in self.DOCUMENT_EXTENSIONS:
            return "file"
        if ext in self.AUDIO_EXTENSIONS or ext in self.VIDEO_EXTENSIONS:
            return "audio_video"
        return None

    def _remove_tokens(self, raw: str, tokens: Sequence[str]) -> str:
        result = raw
        for token in tokens:
            replacements = (
                token,
                f'"{token}"',
                f"'{token}'",
            )
            for item in replacements:
                result = result.replace(item, " ")
        return " ".join(result.split()).strip()

    @staticmethod
    def _dedupe(items: Sequence[str]) -> List[str]:
        seen = set()
        output = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    @staticmethod
    def _build_path_info(path_str: str) -> Dict[str, Any]:
        exists = os.path.exists(path_str)
        info: Dict[str, Any] = {
            "path": path_str,
            "exists": exists,
            "absolute": os.path.abspath(path_str) if exists else path_str,
            "extension": Path(path_str).suffix.lower(),
        }
        if exists:
            info["size"] = os.path.getsize(path_str)
        return info

    @staticmethod
    def _has_existing_path(items: Sequence[Dict[str, Any]]) -> bool:
        return any(bool(item.get("exists")) for item in items)


class DataRouter:
    """
    Unified router for all input modalities.

    Route targets map to the agent modules under `project/agents`.
    """

    ROUTE_MAP = {
        "text": "agents.perception.TextPerceptionAgent",
        "image": "agents.perception.ImagePerceptionAgent",
        "multimodal": "agents.perception.ImagePerceptionAgent",
        "file": "agents.perception.DocumentPerceptionAgent",
        "audio_video": "agents.perception.AudioPerceptionAgent",
    }

    def __init__(self):
        self.classifier = InputClassifier()

    def route(self, input_data: str) -> Dict[str, Any]:
        classification = self.classifier.classify(input_data)
        mode = classification["mode"]
        route_target = self.ROUTE_MAP.get(mode)
        if not route_target:
            return {"error": f"unsupported mode: {mode}", "classification": classification}

        payload: Dict[str, Any]
        if mode == "text":
            payload = {"text": classification["text_content"]}
        elif mode in {"image", "multimodal"}:
            payload = {
                "text": classification["text_content"],
                "images": classification["image_paths"],
            }
        elif mode == "file":
            payload = {
                "documents": classification["file_paths"],
                "text": classification["text_content"],
            }
        else:
            payload = {
                "media": classification["audio_video_paths"],
                "text": classification["text_content"],
            }

        return {
            "route": mode,
            "target_agent": route_target,
            "payload": payload,
            "metadata": classification,
        }


def classify_input(input_str: str) -> Dict[str, Any]:
    return InputClassifier().classify(input_str)

