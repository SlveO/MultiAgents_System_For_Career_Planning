from __future__ import annotations

from typing import Any, Dict, List

try:
    from project.core.schemas import PerceptionResult
except ImportError:
    from core.schemas import PerceptionResult


class MultiModalFusion:
    """Fuses structured PerceptionResult objects from multiple modalities into
    a single LLM-ready context prompt with confidence-weighted ordering and
    cross-modal fact deduplication."""

    @staticmethod
    def fuse(results: List[PerceptionResult]) -> str:
        if not results:
            return ""

        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        blocks: List[str] = []
        seen_facts: set = set()

        for r in sorted_results:
            if r.confidence <= 0.0:
                continue

            identity = (r.subject, r.verification, r.source_id)
            unique_facts = [f for f in r.facts if (*identity, f) not in seen_facts]
            seen_facts.update((*identity, fact) for fact in unique_facts)

            parts = [f"[{r.modality}] confidence={r.confidence:.2f} subject={r.subject} verification={r.verification}"]
            if r.subject in {"job", "reference", "unknown"}:
                parts.append("  主体不是已确认的学生能力；仅作岗位或参考信息。")
            if r.requires_confirmation:
                parts.append("  以下为模型识别的待核实内容，不能当作用户已确认事实；应先向用户确认。")
            if r.summary:
                parts.append(f"  summary: {r.summary}")
            if unique_facts:
                parts.append(f"  facts: {'; '.join(unique_facts[:8])}")
            if r.missing_info:
                parts.append(f"  missing: {'; '.join(r.missing_info[:4])}")

            blocks.append("\n".join(parts))

        return "\n\n".join(blocks)

    @staticmethod
    def fuse_compact(results: List[PerceptionResult]) -> Dict[str, Any]:
        """Return a compact dict suitable for prompt templating."""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        all_facts: List[str] = []
        all_missing: List[str] = []
        modalities: List[str] = []
        material_facts = []

        for r in sorted_results:
            if r.confidence <= 0.0:
                continue
            modalities.append(r.modality)
            if r.subject == "student" and r.verification != "unverified" and not r.requires_confirmation:
                all_facts.extend(r.facts)
            material_facts.append({"subject": r.subject, "verification": r.verification,
                                   "source_id": r.source_id, "facts": r.facts,
                                   "requires_confirmation": r.requires_confirmation})
            all_missing.extend(r.missing_info)

        return {
            "modalities": list(dict.fromkeys(modalities)),
            "facts": list(dict.fromkeys(all_facts))[:12],
            "material_facts": material_facts,
            "missing": list(dict.fromkeys(all_missing))[:6],
            "top_confidence": sorted_results[0].confidence if sorted_results else 0.0,
        }
