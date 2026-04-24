from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

try:
    from .assistant_schemas import EvidenceItem, PerceptionResult
except ImportError:
    from project.assistant_schemas import EvidenceItem, PerceptionResult


def _load_text_processor():
    try:
        from .text import TextProcessor
    except Exception:
        from project.text import TextProcessor
    return TextProcessor


def _load_image_processor():
    try:
        from .image import ImageProcessor
    except Exception:
        from project.image import ImageProcessor
    return ImageProcessor


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


class TextPerceptionAgent:
    def __init__(self, model_path: str):
        TextProcessor = _load_text_processor()
        self.processor = TextProcessor(model_path=model_path)

    def perceive(self, user_text: str) -> PerceptionResult:
        prompt = f"""
请从以下用户输入中抽取结构化信息，并输出严格 JSON：
{{
  "summary": "一句话总结",
  "facts": ["事实1", "事实2"],
  "evidence": ["引用原文片段1", "引用原文片段2"],
  "confidence": 0.0,
  "missing_info": ["缺失信息1"]
}}

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
            confidence=0.65 if facts else 0.45,
            missing_info=[] if facts else ['图像信息不足，建议补充文本上下文'],
            raw_output=raw,
        )

    def unload(self) -> None:
        self.processor.unload()


class DocumentPerceptionAgent:
    @staticmethod
    def _read_txt(path: Path) -> str:
        return path.read_text(encoding='utf-8', errors='ignore')

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return '未安装 pypdf，无法解析 PDF。'

        reader = PdfReader(str(path))
        chunks = []
        for page in reader.pages[:10]:
            chunks.append(page.extract_text() or '')
        return '\n'.join(chunks)

    def perceive(self, document_path: str) -> PerceptionResult:
        path = Path(document_path)
        if not path.exists():
            return PerceptionResult(
                modality='document',
                summary=f'文档不存在: {document_path}',
                facts=[],
                evidence=[],
                confidence=0.0,
                missing_info=['请检查文档路径是否正确'],
                raw_output='',
            )

        suffix = path.suffix.lower()
        if suffix in {'.txt', '.md'}:
            text = self._read_txt(path)
        elif suffix == '.pdf':
            text = self._read_pdf(path)
        else:
            text = '当前仅支持 .txt/.md/.pdf 文档解析。'

        facts = _extract_facts_fallback(text, limit=8)
        return PerceptionResult(
            modality='document',
            summary=f'已解析文档 {path.name}',
            facts=facts,
            evidence=[EvidenceItem(source=document_path, quote=text[:300])],
            confidence=0.6 if facts else 0.3,
            missing_info=[] if facts else ['文档可提取信息不足，建议提供更完整简历/JD'],
            raw_output=text[:3000],
        )


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
