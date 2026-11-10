"""Deterministic, source-labelled career graphs and portable Markdown reports.

This is a projection of stored input, retrieved knowledge and model suggestions.
It does not infer new factual relationships or verify knowledge on the network.
"""
from __future__ import annotations

import hashlib
import html
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .intake import field_value
from .privacy import redact_text
from .schemas import CareerPlanResponse


_FIELD_LABELS = {
    "education": "学历", "major": "专业", "skills": "已掌握技能",
    "interests": "兴趣", "target_role": "意向岗位", "time_budget": "每周时间",
    "preference": "偏好", "constraints": "限制",
}
_RELATION_LABELS = {
    "has_profile": "画像信息", "suggests_role": "建议目标", "has_gap": "待补足能力",
    "has_milestone": "建议阶段", "has_deliverable": "建议交付物", "has_metric": "检查指标",
    "suggests_resource": "建议资源", "suggests_action": "建议行动", "has_risk": "风险提示",
    "requires_skill": "条目所列技能", "lists_resource": "条目所列资源",
    "lists_major": "条目所列专业", "lists_transition": "条目所列发展路径",
    "has_salary_reference": "条目薪资参考", "has_open_question": "待确认",
    "retrieved_knowledge": "检索命中（未逐项核实）",
}
_STATUS_LABELS = {
    "confirmed_input": "有原文证据的画像信息（未经外部核验）",
    "profile_unconfirmed": "画像记录（缺少本次可追溯原文）",
    "unverified_input": "学生材料提取（有原文，尚未经用户确认或外部核验）",
    "model_suggestion": "模型规划建议（非已核实事实）",
    "source_lead_only": "知识库来源线索（字段未逐项核实）",
    "unsourced": "知识库条目（缺少来源且未逐项核实）",
    "not_loaded": "知识命中记录（完整条目未载入）",
}


def _public_text(value: Any) -> str:
    text = redact_text(str(value or ""))
    # Graph/report exports do not retain private file locations from materials.
    # Keep HTTP URLs intact even when their public path contains /srv or a
    # Windows-style drive. Only non-URL spans are considered local paths.
    spans = re.split(r"(https?://[^\s<>]+)", text, flags=re.IGNORECASE)
    for index in range(0, len(spans), 2):
        spans[index] = re.sub(r"[A-Za-z]:[\\/][^\s，。；;、|<>]+", "[REDACTED_PATH]", spans[index])
        spans[index] = re.sub(
            r"/(?:home|Users|tmp|root|mnt|media|var|opt|workspace|srv)(?=/|$|[\s，。；;、|<>])(?:/[^\s，。；;、|<>]*)?",
            "[REDACTED_PATH]", spans[index],
        )
    return "".join(spans)


def _public_plan_id(value: str) -> str:
    # Opaque UUID hex is a machine identifier, not free text. Text redaction can
    # accidentally mask phone-like digit runs inside a valid UUID and break
    # identity checks. Other/custom IDs are hashed to avoid exporting PII/paths.
    if re.fullmatch(r"[0-9a-fA-F]{32}", value):
        return value
    return "plan_" + hashlib.sha256(value.encode()).hexdigest()


def _safe_urls(value: Any) -> list[str]:
    candidates = value if isinstance(value, (list, tuple)) else re.split(r"[;；\s]+", str(value or ""))
    result = []
    for candidate in candidates:
        try:
            parsed = urlsplit(str(candidate))
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                continue
            if parsed.username or parsed.password:
                continue
            cleaned = urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, "", ""))
        except ValueError:
            continue
        if cleaned not in result:
            result.append(_public_text(cleaned))
    return result


def _source_leads(value: Any) -> list[str]:
    """Keep useful textual citations as well as safe URLs, without upgrading trust."""
    candidates = value if isinstance(value, (list, tuple)) else re.split(r"[;；]", str(value or ""))
    leads = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if re.match(r"(?i)^https?://", text):
            urls = _safe_urls([text])
            if not urls:
                continue
            text = urls[0]
        elif re.match(r"(?i)^(?:[a-z][a-z0-9+.-]*://|https?:|javascript:|data:|file:)", text):
            continue
        else:
            # Text citations may contain an embedded source URL. Strip its query
            # and fragment just as for standalone URLs, keeping the citation.
            def clean_url(match):
                safe = _safe_urls([match.group(0)])
                return safe[0] if safe else "[INVALID_SOURCE_URL]"
            text = re.sub(r"https?://[^\s<>]+", clean_url, text, flags=re.IGNORECASE)
        text = _public_text(text)
        meaningful = re.sub(r"\[(?:REDACTED_[A-Z_]+|INVALID_SOURCE_URL)\]", "", text)
        if not re.search(r"[\w\u4e00-\u9fff]", meaningful):
            continue
        if text not in leads:
            leads.append(text)
    return leads


def _values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    # Complete KB entries retain list fields. Retrieved compatibility dictionaries
    # use a comma-delimited representation, which is also accepted for export.
    return [item.strip() for item in re.split(r"[,，;；]", str(value or "")) if item.strip()]


def build_career_graph(
    response: CareerPlanResponse,
    knowledge_items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an offline graph where every node and edge cites provenance IDs."""
    nodes: dict[tuple[str, str], dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    provenance: dict[str, dict] = {}

    def reference(kind: str, key: str, status: str, **extra) -> str:
        ref_id = "p_" + hashlib.sha256(f"{kind}\n{key}".encode()).hexdigest()[:20]
        provenance.setdefault(ref_id, {
            "id": ref_id, "kind": kind, "reference": _public_text(key),
            "status": status, "notice": _STATUS_LABELS[status], **extra,
        })
        return ref_id

    def node(kind: str, label: Any, refs: list[str]) -> str:
        label = _public_text(label).strip()
        key = (kind, label)
        node_id = "n_" + hashlib.sha256(f"{kind}\n{label}".encode()).hexdigest()[:20]
        entry = nodes.setdefault(key, {"id": node_id, "kind": kind, "label": label, "provenance_ids": []})
        entry["provenance_ids"] = list(dict.fromkeys(entry["provenance_ids"] + refs))
        return node_id

    def edge(start: str, end: str, relation: str, refs: list[str]) -> None:
        key = (start, end, relation)
        edge_id = "e_" + hashlib.sha256("\n".join(key).encode()).hexdigest()[:20]
        entry = edges.setdefault(key, {
            "id": edge_id, "from": start, "to": end, "relation": relation,
            "label": _RELATION_LABELS[relation], "provenance_ids": [],
        })
        entry["provenance_ids"] = list(dict.fromkeys(entry["provenance_ids"] + refs))

    plan_ref = reference("planning_output", "plan:structured_output", "model_suggestion")
    plan_node = node("plan", "职业发展规划（建议）", [plan_ref])
    evidence = {item.evidence_id: item for item in response.profile.guidance.evidence}
    for field, label in _FIELD_LABELS.items():
        state = response.profile.guidance.fields.get(field)
        if state is None:
            value = field_value(response.profile, field)
            if value:
                ref = reference("profile_record", f"profile:{field}", "profile_unconfirmed")
                value_node = node("profile", f"{label}：{value}", [ref])
                edge(plan_node, value_node, "has_profile", [ref])
            continue
        if state.status not in {"known", "explicit_none"}:
            continue
        supported = [item for item in state.evidence_ids if item in evidence]
        original = [item for item in supported if not evidence[item].origin.startswith("stored_profile:")]
        # A synthesized stored-profile excerpt does not establish original
        # evidence. Mixed old/new values remain conservatively unconfirmed.
        status = "confirmed_input" if original and len(original) == len(state.evidence_ids) else "profile_unconfirmed"
        if any(evidence[item].subject == "student" and evidence[item].verification == "unverified"
               for item in supported):
            status = "unverified_input"
        refs = [reference("input_evidence" if status == "confirmed_input" else "profile_record",
                          f"guidance:{field}:{item}", status,
                          subject=evidence[item].subject, verification=evidence[item].verification,
                          source_id=evidence[item].source_id) for item in supported]
        if not refs:
            refs = [reference("profile_record", f"profile:{field}", status)]
        values = state.values if state.status == "known" else ["明确没有"]
        for value in values:
            if str(value).strip():
                value_node = node("profile", f"{label}：{value}", refs)
                edge(plan_node, value_node, "has_profile", refs)

    for field, label, suffix in [
        ("financial_budget_cny", "资金预算", "元"), ("city", "城市", ""),
        ("preferred_industries", "行业偏好", ""),
    ]:
        value = getattr(response.profile.constraints, field)
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, list):
            value = "、".join(value)
        ref = reference("profile_record", f"profile:constraints:{field}", "profile_unconfirmed")
        value_node = node("profile_constraint", f"{label}：{value}{suffix}", [ref])
        edge(plan_node, value_node, "has_profile", [ref])

    for attribute, kind, relation in [
        ("target_roles", "role_suggestion", "suggests_role"),
        ("gap_analysis", "gap_suggestion", "has_gap"),
        ("learning_resources", "resource_suggestion", "suggests_resource"),
        ("next_actions", "action_suggestion", "suggests_action"),
        ("risk_flags", "risk", "has_risk"),
        ("follow_up_questions", "open_question", "has_open_question"),
    ]:
        for index, value in enumerate(getattr(response, attribute)):
            if not value.strip():
                continue
            ref = reference("planning_field", f"plan:{attribute}:{index}", "model_suggestion")
            value_node = node(kind, value, [ref])
            edge(plan_node, value_node, relation, [ref])

    for milestone in response.roadmap_30_90_180:
        ref = reference("planning_field", f"plan:roadmap:{milestone.period}", "model_suggestion")
        period = {"30d": "30天", "90d": "90天", "180d": "180天"}[milestone.period]
        milestone_node = node("milestone", f"{period}：{milestone.objective}", [ref])
        edge(plan_node, milestone_node, "has_milestone", [ref])
        for attribute, kind, relation in [
            ("deliverables", "deliverable", "has_deliverable"),
            ("metrics", "metric", "has_metric"),
        ]:
            for index, value in enumerate(getattr(milestone, attribute)):
                ref = reference("planning_field", f"plan:roadmap:{milestone.period}:{attribute}:{index}", "model_suggestion")
                value_node = node(kind, value, [ref])
                edge(milestone_node, value_node, relation, [ref])

    loaded_ids = set()
    hit_ids = set(response.knowledge_hit_ids)
    for item in knowledge_items or []:
        item_id = str(item.get("item_id", item.get("id", "")))
        if not item_id or item_id not in hit_ids:
            continue
        loaded_ids.add(item_id)
        sources = _source_leads(item.get("sources", []))
        urls = _safe_urls(sources)
        ref = reference(
            "knowledge_item", item_id, "source_lead_only" if sources else "unsourced",
            sources=sources, urls=urls, verification_status="not_verified",
        )
        role_node = node("knowledge_role", item.get("role", item_id), [ref])
        edge(plan_node, role_node, "retrieved_knowledge", [ref])
        for field, kind, relation in [
            ("skills", "knowledge_skill", "requires_skill"),
            ("resources", "knowledge_resource", "lists_resource"),
            ("suitable_majors", "knowledge_major", "lists_major"),
            ("transition_paths", "knowledge_transition", "lists_transition"),
        ]:
            for value in _values(item.get(field)):
                value_node = node(kind, value, [ref])
                edge(role_node, value_node, relation, [ref])
        if item.get("salary_hint"):
            salary_node = node("salary_reference", item["salary_hint"], [ref])
            edge(role_node, salary_node, "has_salary_reference", [ref])
    for item_id in sorted(hit_ids - loaded_ids):
        reference("knowledge_item", item_id, "not_loaded")

    return {
        "schema_version": "career-graph-v1",
        "plan_id": _public_plan_id(response.plan_id),
        "nodes": list(nodes.values()), "edges": list(edges.values()),
        "provenance": list(provenance.values()),
        "limitations": [
            "图谱按现有材料、知识条目和规划字段投影生成，不包含额外事实推断。",
            "知识库来源链接仅为核验线索，不代表技能、资源、路径或薪资已经逐项核实。",
            "模型建议与知识库条目分别标记；相同岗位名称不表示建议已被外部证据验证。",
        ],
    }


def _mermaid_label(value: Any) -> str:
    text = " ".join(_public_text(value).split())
    entities = {"&": "#38;", '"': "#quot;", "<": "#60;", ">": "#62;", "|": "#124;", "`": "#96;"}
    return "".join(entities.get(char, char) for char in text)


def render_mermaid(graph: Mapping[str, Any]) -> str:
    """Render labels as inert quoted text; never interpolate supplied graph IDs."""
    lines = ["flowchart TD"]
    ids = {}
    for index, entry in enumerate(graph.get("nodes", [])):
        node_id = f"n{index}"
        ids[str(entry["id"])] = node_id
        lines.append(f'    {node_id}["{_mermaid_label(entry.get("label", ""))}"]')
    for entry in graph.get("edges", []):
        start, end = ids.get(str(entry.get("from"))), ids.get(str(entry.get("to")))
        if start and end:
            label = _mermaid_label(entry.get("label", entry.get("relation", "")))
            lines.append(f'    {start} -->|"{label}"| {end}')
    return "\n".join(lines)


def _markdown_text(value: Any) -> str:
    text = html.escape(_public_text(value), quote=False)
    # Content remains readable text, rather than active images, links or fences
    # that a generated suggestion could inject into an exported document.
    return re.sub(r"([\\`\[\]!|#])", r"\\\1", text)


def _markdown_id(value: str) -> str:
    # Only machine fields use this path. A coincidentally similar user label is
    # still ordinary text and is processed by _markdown_text/redact_text.
    if re.fullmatch(r"(?:[nep]_[0-9a-f]{20}|plan_[0-9a-f]{64}|[0-9a-fA-F]{32})", value):
        return value
    return _markdown_text(value)


def render_report(
    response: CareerPlanResponse,
    graph: Mapping[str, Any],
    display_text: str | None = None,
) -> str:
    """Export text, graph and provenance together without persisting private input."""
    if graph.get("plan_id") != _public_plan_id(response.plan_id):
        raise ValueError("Graph belongs to a different planning output")
    lines = ["# 职业发展规划", "", "以下内容为规划建议；知识来源和未核验边界列在报告末尾。", ""]
    selected_text = display_text if display_text is not None else response.user_facing_advice
    lines.extend([_markdown_text(selected_text), ""])
    profile_nodes = [node for node in graph.get("nodes", []) if node.get("kind") in {"profile", "profile_constraint"}]
    if profile_nodes:
        lines.extend(["## 画像与约束", "", *[f"- {_markdown_text(node['label'])}" for node in profile_nodes], ""])
    lines.extend(["## 结构化规划", ""])
    for label, values in [
        ("目标岗位", response.target_roles), ("能力差距", response.gap_analysis),
        ("学习资源", response.learning_resources), ("下一行动", response.next_actions),
        ("风险提示", response.risk_flags), ("待确认问题", response.follow_up_questions),
    ]:
        if values:
            lines.extend([f"### {label}", "", *[f"- {_markdown_text(value)}" for value in values], ""])
    for milestone in response.roadmap_30_90_180:
        period = {"30d": "30天", "90d": "90天", "180d": "180天"}[milestone.period]
        lines.extend([f"### {period}路线", "", _markdown_text(milestone.objective), ""])
        lines.extend(f"- 交付物：{_markdown_text(value)}" for value in milestone.deliverables)
        lines.extend(f"- 检查指标：{_markdown_text(value)}" for value in milestone.metrics)
        lines.append("")
    lines.extend(["## 职业关系图", "", "```mermaid", render_mermaid(graph), "```", "", "## 来源与核验边界", ""])
    for record in graph.get("provenance", []):
        lines.append(
            f"- {_markdown_id(record['id'])}：{_markdown_text(record.get('reference', ''))}；"
            f"{_markdown_text(record.get('notice', ''))}"
        )
        for source in record.get("sources", record.get("urls", [])):
            lines.append(f"  - {_markdown_text(source)}")
    lines.extend(["", "### 图谱对应记录", ""])
    # The diagrams are compact, while this index keeps every displayed entity and
    # relationship traceable to its precise input, knowledge or planning record.
    for node in graph.get("nodes", []):
        refs = "、".join(_markdown_id(ref) for ref in node.get("provenance_ids", []))
        lines.append(f"- 节点 {_markdown_id(node['id'])}：{_markdown_text(node['label'])}；依据 {refs}")
    for edge in graph.get("edges", []):
        refs = "、".join(_markdown_id(ref) for ref in edge.get("provenance_ids", []))
        lines.append(
            f"- 关系 {_markdown_id(edge['id'])}：{_markdown_id(edge['from'])} → "
            f"{_markdown_id(edge['to'])}（{_markdown_text(edge['label'])}）；依据 {refs}"
        )
    lines.extend(["", *[f"- {_markdown_text(value)}" for value in graph.get("limitations", [])], ""])
    return "\n".join(lines)
