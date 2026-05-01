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

            unique_facts = [f for f in r.facts if f not in seen_facts]
            seen_facts.update(unique_facts)

            parts = [f"[{r.modality}] confidence={r.confidence:.2f}"]
            if r.summary:
                parts.append(f"  summary: {r.summary}")
            if unique_facts:
                parts.append(f"  facts: {'; '.join(unique_facts[:8])}")
            if r.missing_info:
                parts.append(f"  missing: {'; '.join(r.missing_info[:4])}")

            blocks.append("\n".join(parts))

        merged_facts = sorted(seen_facts)[:15]
        if merged_facts:
            blocks.insert(
                0,
                f"[merged_facts] {'; '.join(merged_facts)}",
            )

        return "\n\n".join(blocks)

    @staticmethod
    def fuse_compact(results: List[PerceptionResult]) -> Dict[str, Any]:
        """Return a compact dict suitable for prompt templating."""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        all_facts: List[str] = []
        all_missing: List[str] = []
        modalities: List[str] = []

        for r in sorted_results:
            if r.confidence <= 0.0:
                continue
            modalities.append(r.modality)
            all_facts.extend(r.facts)
            all_missing.extend(r.missing_info)

        return {
            "modalities": list(dict.fromkeys(modalities)),
            "facts": list(dict.fromkeys(all_facts))[:12],
            "missing": list(dict.fromkeys(all_missing))[:6],
            "top_confidence": sorted_results[0].confidence if sorted_results else 0.0,
        }
