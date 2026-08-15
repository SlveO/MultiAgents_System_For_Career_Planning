from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from .schemas import UserProfile


@dataclass(frozen=True)
class FollowUpQuestion:
    key: str
    prompt: str


FOLLOW_UP_QUESTIONS = (
    FollowUpQuestion("education", "1/8 你的年级或最高学历是什么？"),
    FollowUpQuestion("major", "2/8 你的专业或主要学习方向是什么？"),
    FollowUpQuestion("skills", "3/8 你已经掌握哪些技能？请用逗号分隔。"),
    FollowUpQuestion("interests", "4/8 你感兴趣的工作内容或领域是什么？"),
    FollowUpQuestion("target_role", "5/8 你目前最想尝试的目标岗位是什么？"),
    FollowUpQuestion("time_budget", "6/8 你每周可稳定投入多少小时？"),
    FollowUpQuestion("preference", "7/8 你偏好的城市或行业是什么？"),
    FollowUpQuestion("constraints", "8/8 你当前最大的限制或短板是什么？"),
)


def collect_follow_up_answers(
    input_fn: Callable[[str], str] = input,
) -> Dict[str, str]:
    answers: Dict[str, str] = {}
    for question in FOLLOW_UP_QUESTIONS:
        answers[question.key] = (input_fn(f"{question.prompt}\n> ") or "").strip()
    return answers


def _split_items(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、;；]", value or "") if item.strip()]


def _parse_hours(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    return int(match.group()) if match else None


def apply_answers_to_profile(
    profile: UserProfile,
    answers: Mapping[str, str],
) -> UserProfile:
    updated = profile.model_copy(deep=True)
    education = (answers.get("education") or "").strip()
    major = (answers.get("major") or "").strip()
    skills = _split_items(answers.get("skills", ""))
    interests = _split_items(answers.get("interests", ""))
    target_role = (answers.get("target_role") or "").strip()
    preference = (answers.get("preference") or "").strip()
    main_constraints = _split_items(answers.get("constraints", ""))

    if education:
        updated.education_stage = education
        updated.current_stage = education
        updated.constraints.education_level = education
    if major:
        updated.major = major
    if skills:
        updated.skills = list(dict.fromkeys(updated.skills + skills))
    if interests:
        updated.interests = list(dict.fromkeys(updated.interests + interests))
    if target_role:
        updated.target_role = target_role
    if preference:
        updated.preference = preference
    if main_constraints:
        updated.main_constraints = list(
            dict.fromkeys(updated.main_constraints + main_constraints)
        )
    hours = _parse_hours(answers.get("time_budget", ""))
    if hours is not None:
        updated.constraints.time_budget_hours_per_week = hours
    return updated
