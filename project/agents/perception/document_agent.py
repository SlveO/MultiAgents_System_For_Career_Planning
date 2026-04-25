from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:
    from project.core.schemas import EvidenceItem, PerceptionResult
except ImportError:
    from core.schemas import EvidenceItem, PerceptionResult

from .base import _extract_facts_fallback


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
