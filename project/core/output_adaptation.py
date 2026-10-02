"""Adapt expression using guarded references to an immutable career plan."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from .schemas import CareerPlanResponse, FeedbackAdaptationResult, FeedbackLayout, OutputVersion


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
    facts = response.model_dump(exclude={"plan_id", "output_version", "user_facing_advice"})
    return text_hash(json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def original_version(response: CareerPlanResponse) -> OutputVersion:
    return OutputVersion(
        plan_id=response.plan_id, version=1, display_text=response.user_facing_advice,
        facts_sha256=facts_hash(response), source_text_sha256=text_hash(response.user_facing_advice),
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


def adapt_output(response: CareerPlanResponse, feedback: str, request_id: str, model: FeedbackModel) -> FeedbackAdaptationResult:
    baseline = response.output_version
    if baseline is None:
        raise ValueError("Original output version is required")
    started = time.perf_counter()
    result = FeedbackAdaptationResult(
        request_id=request_id, session_id=response.session_id, plan_id=response.plan_id,
        feedback=feedback, status="unchanged", output_version=baseline.model_copy(deep=True),
        reason="suitable",
    )
    if baseline.facts_sha256 != facts_hash(response) or baseline.source_text_sha256 != text_hash(response.user_facing_advice):
        result.status, result.reason = "failed", "source_changed"
        return result
    if feedback == "合适":
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
        }))
        result.model_used = True
        text, selected = render_layout(layout, catalog, required)
        if (baseline.facts_sha256 != facts_hash(response)
                or baseline.source_text_sha256 != text_hash(response.user_facing_advice)):
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
