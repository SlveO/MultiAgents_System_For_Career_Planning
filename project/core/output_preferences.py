"""Render initial displays from explicit user preferences and source references.

This is deterministic presentation selection. It does not measure cognitive
load, infer a student's actual abilities, or implement a VAE-based model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from .schemas import CareerPlanResponse, OutputVersion


class OutputPreferences(BaseModel):
    """User-declared expression preferences, separate from factual profiles."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    detail: Literal["brief", "standard", "detailed"] = "standard"
    knowledge_level: Literal["unspecified", "beginner", "familiar"] = "unspecified"
    presentation: Literal["bullets", "paragraph"] = "bullets"
    user_reported: Literal[True] = True
    source: Literal["cli", "user_input", "saved_preferences"] = "cli"

    @field_validator("user_reported", mode="before")
    @classmethod
    def require_explicit_user_report(cls, value):
        if value is not True:
            raise ValueError("Output preferences must be declared by the user")
        return value


def preferences_prompt_hint(preferences: OutputPreferences | None) -> str:
    """Describe expression choices without elevating them into career facts."""
    if preferences is None:
        return ""
    preferences = OutputPreferences.model_validate(preferences)
    detail = {"brief": "简要，突出核心目标与下一行动", "standard": "标准详略，给出必要解释与资源",
              "detailed": "详细，展开已有依据支持的路线、交付物与检查指标"}[preferences.detail]
    background = {
        "unspecified": "用户未说明讲解背景，不推测其知识或技能水平",
        "beginner": "用户选择入门讲解，解释必要术语；这不证明其实际技能不足",
        "familiar": "用户选择熟悉术语的讲解方式；这不证明其已经掌握相关技能",
    }[preferences.knowledge_level]
    presentation = {"bullets": "分点呈现", "paragraph": "分段呈现"}[preferences.presentation]
    return (
        f"用户自报的表达偏好（来源：{preferences.source}）：{detail}；{presentation}；{background}。"
        "这些是呈现偏好，不是已核实的能力或职业事实，也不是认知负荷测量。"
        "保持已确认的目标、时间、资金预算、约束和风险，不因详略选择新增事实。"
    )


def render_initial_output(
    response: CareerPlanResponse,
    preferences: OutputPreferences | None,
) -> OutputVersion:
    """Choose original catalog references for version one without rewriting.

    Missing preferences preserve the existing initial display. Configured
    preferences select essential entries, a bounded middle selection, or the
    complete catalog. Required content is retained at every detail setting.
    """
    # Kept lazy so schemas.py may import OutputPreferences without a cycle.
    from .output_adaptation import brief_reading_view, build_catalog, original_version, render_layout
    from .privacy import redact_text
    from .schemas import FeedbackLayout

    original = original_version(response)
    if preferences is None:
        return original
    preferences = OutputPreferences.model_validate(preferences)
    if preferences.detail == "brief":
        return original.model_copy(update={"reading_view": brief_reading_view(response, original.display_text)})
    catalog = build_catalog(response)
    if not catalog:
        return original
    # Source fields may arrive in any order. Order the local reference catalog
    # chronologically without mutating the original planning response.
    period_order = {"30d": 0, "90d": 1, "180d": 2}
    catalog.sort(key=lambda item: period_order[item.item_id.split(":")[1]]
                 if item.section == "roadmap" else -1)

    required = {item.item_id for item in catalog if item.essential}
    selected = set(required)
    if preferences.detail == "detailed":
        selected.update(item.item_id for item in catalog)
    elif preferences.detail == "standard":
        for section in ("gaps", "resources"):
            # Keep the middle view bounded even when the planner returns many resources.
            section_ids = [item.item_id for item in catalog if item.section == section]
            selected.update(section_ids[:3])
        for period in ("30d", "90d", "180d"):
            for field in ("deliverables", "metrics"):
                prefix = f"roadmap:{period}:{field}:"
                item = next((item for item in catalog if item.item_id.startswith(prefix)), None)
                if item is not None:
                    selected.add(item.item_id)
    if not selected:
        # A sparse plan can contain only prose or optional content. Keep an
        # existing source item rather than fabricate a placeholder or an empty view.
        selected.add(catalog[0].item_id)

    grouped: dict[str, list[str]] = {}
    for item in catalog:
        if item.item_id in selected:
            grouped.setdefault(item.section, []).append(item.item_id)
    order = ("advice", "targets", "gaps", "roadmap", "actions", "resources", "constraints", "risks", "questions")
    layout = FeedbackLayout.model_validate({
        "sections": [
            {"section": section, "item_ids": grouped[section], "style": preferences.presentation}
            for section in order if section in grouped
        ],
    })
    display_text, item_ids = render_layout(layout, catalog, required)
    return original.model_copy(update={"display_text": redact_text(display_text), "item_ids": item_ids})
