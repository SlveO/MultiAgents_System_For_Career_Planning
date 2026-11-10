"""Export a derived, redacted report without overwriting existing files."""
from __future__ import annotations

import json
from pathlib import Path

from .knowledge_graph import build_career_graph, render_mermaid, render_report
from .privacy import redact_data
from .schemas import CareerPlanResponse


def export_report(
    response: CareerPlanResponse,
    knowledge_items: list[dict],
    directory: str | Path,
    *,
    display_text: str | None = None,
) -> Path:
    # The positional argument is retained for existing callers. A historical
    # plan without a snapshot cannot reconstruct its original knowledge safely.
    knowledge_items = response.knowledge_snapshot["hits"] if response.knowledge_snapshot is not None else []
    graph = build_career_graph(response, knowledge_items)
    if response.knowledge_snapshot is None:
        graph.setdefault("limitations", []).append("此历史规划没有知识快照；未使用现行知识库补全当时来源。")
    artifacts = {
        "report.md": render_report(response, graph, display_text=display_text),
        "graph.mmd": render_mermaid(graph),
        "graph.json": json.dumps(redact_data(graph), ensure_ascii=False, indent=2),
    }
    for content in artifacts.values():
        content.encode("utf-8")
    target = Path(directory)
    # The directory itself is reserved atomically; a concurrent invocation or an
    # existing report fails before any artifact is changed.
    target.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    try:
        for name, content in artifacts.items():
            path = target / name
            with path.open("x", encoding="utf-8") as handle:
                created.append(path)
                handle.write(content)
    except (OSError, UnicodeError):
        for path in created:
            path.unlink(missing_ok=True)
        target.rmdir()
        raise
    return target
