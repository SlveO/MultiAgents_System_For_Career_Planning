"""Adapt expression using guarded references to an immutable career plan."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .privacy import redact_data, redact_text
from .schemas import CareerPlanResponse, FeedbackAdaptationResult, FeedbackLayout, OutputVersion, ReadingView


class FeedbackModel(Protocol):
    def analyze(self, context: dict) -> FeedbackLayout: ...
    def unload(self) -> None: ...


@dataclass(frozen=True)
class DisplayItem:
    item_id: str
    section: str
    text: str
    essential: bool = False


_LABELS = {
    "advice": "规划建议", "targets": "目标岗位", "gaps": "能力差距",
    "roadmap": "行动路线", "actions": "下一行动", "resources": "学习资源",
    "constraints": "当前约束", "risks": "注意事项", "questions": "待确认信息",
}
_PERIODS = {"30d": "30天", "90d": "90天", "180d": "180天"}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def facts_hash(response: CareerPlanResponse) -> str:
    # Hash the same privacy-safe projection that survives persistence/restart.
    # Raw quotes and paths are deliberately absent from this representation.
    facts = redact_data(response.model_dump(exclude={"plan_id", "output_version", "user_facing_advice"}))
    return text_hash(json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def original_version(response: CareerPlanResponse) -> OutputVersion:
    source_text = redact_text(response.user_facing_advice)
    return OutputVersion(
        plan_id=response.plan_id, version=1, display_text=source_text,
        facts_sha256=facts_hash(response), source_text_sha256=text_hash(source_text),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def build_catalog(response: CareerPlanResponse) -> list[DisplayItem]:
    items: list[DisplayItem] = []

    def add(item_id, section, text, essential=False):
        if text.strip():
            items.append(DisplayItem(item_id, section, text, essential))

    add("advice", "advice", response.user_facing_advice)
    for field, section, essential in [
        ("target_roles", "targets", True), ("gap_analysis", "gaps", False),
        ("next_actions", "actions", True), ("learning_resources", "resources", False),
        ("risk_flags", "risks", True), ("follow_up_questions", "questions", True),
    ]:
        for index, text in enumerate(getattr(response, field)):
            add(f"{field}:{index}", section, text, essential)
    for milestone in response.roadmap_30_90_180:
        prefix = f"roadmap:{milestone.period}"
        period = _PERIODS[milestone.period]
        add(f"{prefix}:objective", "roadmap", f"{period}目标：{milestone.objective}", True)
        for field, label in [("deliverables", "交付物"), ("metrics", "指标")]:
            for index, text in enumerate(getattr(milestone, field)):
                add(f"{prefix}:{field}:{index}", "roadmap", f"{period}{label}：{text}")
    constraints = response.profile.constraints
    for field, label, suffix in [
        ("time_budget_hours_per_week", "每周时间", "小时"),
        ("financial_budget_cny", "资金预算", "元"),
    ]:
        value = getattr(constraints, field)
        if value is not None:
            add(f"constraints:{field}", "constraints", f"{label}：{value:g}{suffix}", True)
    if constraints.city:
        add("constraints:city", "constraints", f"城市：{constraints.city}", True)
    if response.profile.preference:
        add("constraints:preference", "constraints", f"偏好：{response.profile.preference}", True)
    for field, values, label in [
        ("preferred_industries", constraints.preferred_industries, "行业偏好"),
        ("main_constraints", response.profile.main_constraints, "限制"),
    ]:
        for index, text in enumerate(values):
            add(f"constraints:{field}:{index}", "constraints", f"{label}：{text}", True)
    field_labels = {
        "education": "学历", "major": "专业", "skills": "技能", "interests": "兴趣",
        "target_role": "目标岗位", "time_budget": "时间", "preference": "偏好", "constraints": "限制",
    }
    status_labels = {"missing": "未提供", "unknown": "未知", "refused": "暂不回答", "conflict": "有矛盾待确认"}
    for field, state in response.profile.guidance.fields.items():
        if state.status in status_labels:
            add(f"unresolved:{field}", "constraints", f"{field_labels[field]}：{status_labels[state.status]}", True)
    return items


def render_layout(layout: FeedbackLayout, catalog: list[DisplayItem], required: set[str]) -> tuple[str, list[str]]:
    by_id = {item.item_id: item for item in catalog}
    seen_ids, seen_sections = set(), set()
    parts, selected = [], []
    for section in layout.sections:
        if section.section in seen_sections:
            raise ValueError("duplicate_section")
        seen_sections.add(section.section)
        texts = []
        ids = list(section.item_ids)
        # Timeline rendering retains the original chronology, whatever ordering is proposed.
        if section.section == "roadmap":
            order = {item.item_id: index for index, item in enumerate(catalog)}
            ids.sort(key=lambda item_id: order.get(item_id, -1))
        for item_id in ids:
            if item_id not in by_id:
                raise ValueError("unknown_reference")
            if item_id in seen_ids:
                raise ValueError("duplicate_reference")
            item = by_id[item_id]
            if item.section != section.section:
                raise ValueError("wrong_section")
            seen_ids.add(item_id)
            selected.append(item_id)
            texts.append(item.text)
        body = "\n".join(f"- {text}" for text in texts) if section.style == "bullets" else "\n\n".join(texts)
        parts.append(f"{_LABELS[section.section]}\n{body}")
    if not required <= seen_ids:
        raise ValueError("missing_required_facts")
    return "\n\n".join(parts), selected


def brief_reading_view(response: CareerPlanResponse, full_text: str) -> ReadingView:
    """Select whole source items deterministically, with a 500-code-point soft cap."""
    catalog = build_catalog(response)
    by_id = {item.item_id: item for item in catalog}
    if not any(item.section != "advice" for item in catalog):
        return ReadingView(excerpt=full_text, notice="缺少结构化内容，无法安全摘选，已保留完整正文。")
    required = ["target_roles:0", "next_actions:0", "constraints:time_budget_hours_per_week",
                "constraints:financial_budget_cny", "constraints:city"]
    optional = ["risk_flags:0", "next_actions:1"] + [f"roadmap:{period}:objective" for period in ("30d", "90d", "180d")]
    selected: list[str] = []

    def render(ids: list[str]) -> tuple[str, list[str]]:
        parts, actual = [], []
        for section in ("targets", "actions", "constraints", "risks", "roadmap"):
            texts = set()
            lines = []
            # Preserve source order within a section.
            for item in catalog:
                if item.item_id in ids and item.section == section and item.text not in texts:
                    texts.add(item.text)
                    actual.append(item.item_id)
                    lines.append(f"- {redact_text(item.text)}")
            if lines:
                parts.append(_LABELS[section] + "\n" + "\n".join(lines))
        return "\n\n".join(parts), actual

    selected.extend(item_id for item_id in required if item_id in by_id)
    for item_id in optional:
        if item_id in by_id and len(render(selected + [item_id])[0]) <= 500:
            selected.append(item_id)
    excerpt, actual = render(selected)
    # A structured report without any eligible fields cannot safely be summarized.
    if not excerpt:
        return ReadingView(excerpt=full_text, notice="缺少可摘选的结构化内容，已保留完整正文。")
    notice = "必要原文较长，简要正文超过 500 字符。" if len(excerpt) > 500 else ""
    return ReadingView(excerpt=excerpt, item_ids=actual, notice=notice)


def adapt_output(
    response: CareerPlanResponse,
    feedback: str,
    request_id: str,
    model: FeedbackModel,
    *,
    baseline: OutputVersion | None = None,
) -> FeedbackAdaptationResult:
    """Adapt the latest display while retaining the canonical planning response.

    Callers may supply a persisted later display as ``baseline``. Its hashes
    must still identify the original plan, never a rewritten planning object.
    """
    baseline = baseline or response.output_version
    if baseline is None:
        raise ValueError("Original output version is required")
    started = time.perf_counter()
    result = FeedbackAdaptationResult(
        request_id=request_id, session_id=response.session_id, plan_id=response.plan_id,
        feedback=feedback, status="unchanged", output_version=baseline.model_copy(deep=True),
        source_version=baseline.version, reason="suitable",
    )
    if (baseline.plan_id != response.plan_id
            or baseline.facts_sha256 != facts_hash(response)
            or baseline.source_text_sha256 != text_hash(redact_text(response.user_facing_advice))):
        result.status, result.reason = "failed", "source_changed"
        return result
    if feedback == "合适":
        return result
    if feedback == "过于详细":
        view = brief_reading_view(response, baseline.display_text)
        if baseline.reading_view == view:
            result.reason = "already_brief"
            return result
        result.status, result.reason = "adapted", "brief_view"
        result.output_version = baseline.model_copy(update={
            "version": baseline.version + 1, "parent_version": baseline.version,
            "reading_view": view, "created_at": datetime.now(timezone.utc).isoformat(),
        }, deep=True)
        return result
    if feedback == "过短" and baseline.reading_view is not None:
        result.status, result.reason = "adapted", "full_view"
        result.output_version = baseline.model_copy(update={
            "version": baseline.version + 1, "parent_version": baseline.version,
            "reading_view": None, "created_at": datetime.now(timezone.utc).isoformat(),
        }, deep=True)
        return result
    catalog = build_catalog(response)
    required = {item.item_id for item in catalog if feedback == "过短" or item.essential}
    if not catalog or all(item.section == "advice" for item in catalog):
        result.reason = "no_content_to_adjust"
        return result
    try:
        layout = FeedbackLayout.model_validate(model.analyze({
            "feedback": feedback,
            "catalog": [item.__dict__ for item in catalog],
            "required_ids": sorted(required),
            "source_version": baseline.version,
            "current_display_text": baseline.display_text,
        }))
        result.model_used = True
        text, selected = render_layout(layout, catalog, required)
        text = redact_text(text)
        if (baseline.facts_sha256 != facts_hash(response)
                or baseline.source_text_sha256 != text_hash(redact_text(response.user_facing_advice))):
            raise ValueError("source_changed")
        shorter = len(text.strip()) < len(baseline.display_text.strip())
        longer = len(text.strip()) > len(baseline.display_text.strip())
        if not (longer if feedback == "过短" else shorter):
            result.reason = "length_constraint"
            return result
        result.status, result.reason = "adapted", "expanded" if feedback == "过短" else "compressed"
        result.output_version = OutputVersion(
            plan_id=response.plan_id, version=baseline.version + 1, parent_version=baseline.version,
            display_text=text, facts_sha256=baseline.facts_sha256,
            source_text_sha256=baseline.source_text_sha256, item_ids=selected,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as exc:
        result.status = "failed"
        result.reason = "source_changed" if str(exc) == "source_changed" else "invalid_layout"
    except Exception:
        # Loader errors can contain private paths; persist only a stable category.
        result.status, result.reason = "failed", "model_unavailable"
    finally:
        model.unload()
        result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result
