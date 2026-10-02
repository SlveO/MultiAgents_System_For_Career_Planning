from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from .schemas import AnswerRecord, FieldState, GuidanceEvidence, UserProfile


@dataclass(frozen=True)
class FollowUpQuestion:
    key: str
    prompt: str


FOLLOW_UP_QUESTIONS = (
    FollowUpQuestion("education", "你的年级或最高学历是什么？"),
    FollowUpQuestion("major", "你的专业或主要学习方向是什么？"),
    FollowUpQuestion("skills", "你已经掌握哪些技能？请用逗号分隔；没有可回答‘无’。"),
    FollowUpQuestion("interests", "你感兴趣的工作内容或领域是什么？"),
    FollowUpQuestion("target_role", "你目前最想尝试的具体岗位是什么？"),
    FollowUpQuestion("time_budget", "你每周可稳定投入多少小时？请给出一个数值。"),
    FollowUpQuestion("preference", "你偏好的城市或行业是什么？"),
    FollowUpQuestion("constraints", "你当前最大的限制或短板是什么？"),
)
FIELD_PATHS = {
    "education": "education_stage", "major": "major", "skills": "skills",
    "interests": "interests", "target_role": "target_role",
    "time_budget": "constraints.time_budget_hours_per_week",
    "preference": "preference", "constraints": "main_constraints",
}
_UNKNOWN = {"不清楚", "不知道", "未知", "暂不确定", "还没想好"}
_REFUSED = {"不想回答", "暂不回答", "拒答", "跳过"}
_NONE = {"无", "没有", "暂无", "没有技能", "无技能", "没有限制", "无偏好"}


def collect_follow_up_answers(input_fn: Callable[[str], str] = input) -> Dict[str, str]:
    """Legacy eight-field collector; the CLI uses adaptive guidance instead."""
    return {q.key: input_fn(f"{q.prompt}\n> ").strip() for q in FOLLOW_UP_QUESTIONS}


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、;；]", value) if item.strip()]


def _parse_hours(value: str) -> float | None:
    match = re.fullmatch(r"(?:每周\s*)?(\d+(?:\.\d+)?)\s*(?:个?小时|h)?", value.strip(), re.I)
    if not match:
        return None
    hours = float(match.group(1))
    return hours if 0 <= hours <= 168 else None


def answer_status(field: str, value: str) -> str:
    value = value.strip()
    if not value:
        return "missing"
    if value in _UNKNOWN:
        return "unknown"
    if value in _REFUSED:
        return "refused"
    if value in _NONE:
        return "explicit_none"
    if field == "time_budget" and _parse_hours(value) is None:
        return "unknown"
    return "known"


def field_value(profile: UserProfile, field: str) -> str:
    if field == "time_budget":
        hours = profile.constraints.time_budget_hours_per_week
        return "" if hours is None else f"{hours:g}"
    value = getattr(profile, FIELD_PATHS[field])
    if field == "education":
        value = value or profile.constraints.education_level or profile.current_stage
    if field == "preference" and not value:
        value = "、".join(filter(None, [profile.constraints.city, *profile.constraints.preferred_industries]))
    return "、".join(value) if isinstance(value, list) else (value or "")


def set_field(profile: UserProfile, field: str, value: str) -> None:
    if field == "education":
        profile.education_stage = profile.current_stage = value
        profile.constraints.education_level = value or None
    elif field == "time_budget":
        profile.constraints.time_budget_hours_per_week = _parse_hours(value) if value else None
    elif field in {"skills", "interests", "constraints"}:
        setattr(profile, FIELD_PATHS[field], list(dict.fromkeys(_split_items(value))))
    else:
        setattr(profile, FIELD_PATHS[field], value)
        if field == "preference" and not value:
            profile.constraints.city = None
            profile.constraints.preferred_industries = []


def add_evidence(profile: UserProfile, origin: str, excerpt: str) -> str:
    evidence_id = hashlib.sha256(f"{origin}\n{excerpt}".encode()).hexdigest()[:20]
    if not any(e.evidence_id == evidence_id for e in profile.guidance.evidence):
        profile.guidance.evidence.append(GuidanceEvidence(
            evidence_id=evidence_id, origin=origin, excerpt=excerpt,
        ))
    return evidence_id


def initialize_fields(profile: UserProfile) -> None:
    for field in FIELD_PATHS:
        if field in profile.guidance.fields:
            continue
        value = field_value(profile, field)
        refs = [add_evidence(profile, f"stored_profile:{field}", value)] if value else []
        profile.guidance.fields[field] = FieldState(
            status="known" if value else "missing", values=[value] if value else [], evidence_ids=refs,
        )
        if value:
            set_field(profile, field, value)


def record_answer(
    profile: UserProfile, field: str, raw: str, *, question_id: str,
    prompt: str = "", round_number: int = 0, origin: str = "answers_json",
) -> None:
    if field not in FIELD_PATHS:
        raise ValueError(f"Unsupported guidance field: {field}")
    initialize_fields(profile)
    value = raw.strip()
    status = answer_status(field, value)
    ref = add_evidence(profile, f"{origin}:{field}", raw)
    profile.guidance.answers.append(AnswerRecord(
        field=field, question_id=question_id, prompt=prompt, raw_answer=raw,
        status=status, round_number=round_number, origin=origin, evidence_id=ref,
    ))
    old = profile.guidance.fields[field]
    if old.status == "conflict" and status in {"missing", "unknown", "refused"}:
        old.evidence_ids = list(dict.fromkeys([*old.evidence_ids, ref]))
        return
    profile.guidance.fields[field] = FieldState(
        status=status, values=[value] if status in {"known", "explicit_none"} else [], evidence_ids=[ref],
        resolved_evidence_ids=[e.evidence_id for e in profile.guidance.evidence],
    )
    # Explicit user updates replace old facts, including clearing a formerly known value.
    set_field(profile, field, value if status == "known" else "")


def apply_answers_to_profile(profile: UserProfile, answers: Mapping[str, str]) -> UserProfile:
    updated = profile.model_copy(deep=True)
    initialize_fields(updated)
    for field, raw in answers.items():
        record_answer(updated, field, raw, question_id=f"provided:{field}")
    return updated
