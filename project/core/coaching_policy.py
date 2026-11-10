"""Explainable coaching rules; readiness thresholds never change input facts.

This module describes grounded conversation actions for the existing eight-field
profile. It neither measures emotion/cognitive load nor learns a dialogue policy.
"""
from __future__ import annotations

import hashlib
import json
import re

from .intake import FIELD_PATHS, field_value
from .schemas import FieldState, GuidanceQuestion, UserProfile


_PRIORITY = ("target_role", "skills", "time_budget", "education", "major", "constraints", "preference", "interests")
_WEIGHTS = {"target_role": 3, "skills": 2, "time_budget": 2, "education": 1, "major": 1, "interests": 1, "preference": 1, "constraints": 1}
_CORE_FIELDS = {"target_role", "skills", "time_budget"}
_LABELS = {"target_role": "目标岗位", "skills": "已掌握技能", "time_budget": "每周可投入时间", "education": "学历或年级", "major": "专业", "interests": "工作兴趣", "preference": "城市或行业偏好", "constraints": "现实限制"}
_PROTECTED = {"unknown", "refused", "explicit_none"}
_RESOLVED = {"known", "explicit_none"}
_VAGUE_GOALS = {"职业规划", "职业发展", "就业", "找工作", "找实习", "实习", "求职", "工作", "就业方向", "随便", "都可以"}
_MISSING_PROMPTS = {
    "target_role": "你目前最愿意先尝试哪一个具体岗位？如果暂时不确定，可以回答不知道。",
    "skills": "你已经掌握哪些可用于目标工作的技能？可举一个课程或项目中的例子，没有可回答无。",
    "time_budget": "考虑课程和生活安排，你每周能稳定投入多少小时？请给出一个数值。",
    "education": "你目前的年级或最高学历是什么？",
    "major": "你的专业或主要学习方向是什么？",
    "constraints": "为了让规划可执行，你希望先考虑哪项现实限制？没有可回答无。",
    "preference": "选择城市或行业时，你最看重什么？请说明首选及原因，没有偏好可回答无。",
    "interests": "哪种工作内容让你愿意持续投入？可举一个课程或项目中的例子，暂不确定可回答不知道。",
}
_RULES = [
    {"action": "resolve_conflict", "trigger": "同一字段仍有未解决矛盾，且用户未拒绝继续澄清", "purpose": "请用户确认当前情况，不由模型裁定矛盾"},
    {"action": "clarify_goal", "trigger": "目标岗位缺失或只有宽泛求职方向", "purpose": "将开放目标缩小到可尝试的具体岗位"},
    {"action": "clarify_values", "trigger": "城市/行业偏好或兴趣缺失", "purpose": "通过选择原因和经历澄清偏好，不推断心理特征"},
    {"action": "prioritize", "trigger": "已确认的目标或偏好包含多个选项，且尚未交互澄清", "purpose": "由用户说明优先顺序，保留用户明确保留的其他选项"},
    {"action": "collect_background", "trigger": "技能、时间或学业背景等必要信息缺失", "purpose": "补足可执行规划所需条件"},
]


def _states(profile: UserProfile) -> dict[str, FieldState]:
    result = {}
    for field in FIELD_PATHS:
        if field in profile.guidance.fields:
            result[field] = profile.guidance.fields[field]
        else:
            value = field_value(profile, field)
            result[field] = FieldState(status="known" if value else "missing", values=[value] if value else [])
    return result


def _question_key(state: FieldState, question: GuidanceQuestion) -> str:
    # Match guidance.question_key without importing guidance (which consumes this
    # module). A new prompt wording alone must not make an old question eligible.
    signature = [question.field, question.reason]
    if question.reason != "missing":
        signature.extend(sorted(state.evidence_ids))
        signature.extend(state.values)
    return hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()[:20]


def _multiple_options(values: list[str]) -> bool:
    # C/C++ names a technology family, not two career preferences.
    values = [re.sub(r"\bC\s*/\s*C\+\+", "C与C++", value, flags=re.I) for value in values]
    return len({part.strip() for value in values for part in re.split(r"[、,，;；/]|或者|或", value) if part.strip()}) > 1


def build_coaching_context(profile: UserProfile, *, round_number: int, max_rounds: int) -> dict:
    """Return deterministic actions and readiness metrics without mutating profile.

    ``round_number`` is one-based and includes the question about to be asked.
    ``follow_up_threshold`` is an engineering target for weighted information
    completeness (0.95 early, 0.75 in the last round), not factual confidence.
    Essential gaps and unresolved conflicts take precedence over that target.
    """
    if round_number < 1 or max_rounds < 0:
        raise ValueError("Coaching rounds require round_number >= 1 and max_rounds >= 0")
    states = _states(profile)
    evidence = {item.evidence_id: item for item in profile.guidance.evidence}
    # Synthetic copies of a stored profile do not establish original evidence.
    original_ids = {key for key, value in evidence.items() if value.excerpt.strip() and not value.origin.startswith("stored_profile:")}
    resolved = {field for field, state in states.items() if state.status in _RESOLVED and (any(value.strip() for value in state.values) or state.status == "explicit_none")}
    supported = {field for field in resolved if original_ids.intersection(states[field].evidence_ids)}
    total_weight = sum(_WEIGHTS.values())
    resolved_weight = sum(_WEIGHTS[field] for field in resolved)
    supported_weight = sum(_WEIGHTS[field] for field in supported)
    remaining = max(0, max_rounds - round_number + 1)
    progress = min(1.0, (round_number - 1) / (max_rounds - 1)) if max_rounds > 1 else 1.0
    threshold = round(0.95 - 0.20 * progress, 4)
    metrics = {
        "information_completeness": round(resolved_weight / total_weight, 4),
        "evidence_coverage": round(supported_weight / resolved_weight, 4) if resolved_weight else 0.0,
        "response_coverage": round(sum(state.status != "missing" for state in states.values()) / len(states), 4),
        "follow_up_threshold": threshold,
        "resolved_weight": resolved_weight, "supported_weight": supported_weight, "total_weight": total_weight,
    }
    candidates = []

    def candidate(action, field, reason, prompt, priority):
        state = states[field]
        question = GuidanceQuestion(
            field=field, reason=reason, prompt=prompt,
            evidence_ids=sorted(
                (item for item in state.evidence_ids if item in evidence),
                key=lambda item: item not in original_ids,
            )[:8],
        )
        key = _question_key(state, question)
        answers = [item for item in profile.guidance.answers if item.field == field]
        blocked = None
        if state.status in _PROTECTED:
            blocked = "user_boundary"
        elif key in profile.guidance.asked_keys:
            blocked = "already_asked"
        elif reason == "missing" and answers:
            blocked = "already_answered"
        elif reason == "conflict" and answers and answers[-1].status in {"unknown", "refused"}:
            blocked = "user_boundary"
        elif reason == "clarify" and any(item.origin == "interactive" for item in answers):
            blocked = "already_clarified"
        elif reason == "clarify" and not original_ids.intersection(question.evidence_ids):
            blocked = "no_original_evidence"
        candidates.append({
            "action": action, "focus_field": field, "question": question.model_dump(),
            "question_key": key, "priority": priority, "eligible": blocked is None,
            "blocked_reason": blocked,
        })

    for field in _PRIORITY:
        state = states[field]
        if state.status == "conflict":
            candidate("resolve_conflict", field, "conflict", f"关于{_LABELS[field]}，已有信息不一致。请以现在的情况确认；也可以回答不知道或暂不回答。", 100 + _WEIGHTS[field])
        elif state.status == "missing":
            action = "clarify_goal" if field == "target_role" else "clarify_values" if field in {"interests", "preference"} else "collect_background"
            priority = {"target_role": 90, "skills": 80, "time_budget": 75, "education": 60, "major": 55, "constraints": 35, "preference": 40, "interests": 30}[field]
            candidate(action, field, "missing", _MISSING_PROMPTS[field], priority)
        elif state.status == "known" and field == "target_role":
            values = state.values
            if values and all(value.strip() in _VAGUE_GOALS for value in values):
                candidate("clarify_goal", field, "clarify", "你提到的方向还比较宽泛。如果先做一次尝试，你最愿意从哪一个具体岗位开始？暂不确定可回答不知道。", 95)
            elif _multiple_options(values):
                candidate("prioritize", field, "clarify", "你提到了多个目标岗位。如果先安排一次尝试，请说明首选岗位及其他选项的顺序；暂时无法排序可回答不知道。", 85)
        elif state.status == "known" and field == "preference" and _multiple_options(state.values):
            candidate("prioritize", field, "clarify", "在已提到的城市或行业中，你会如何排序？请说明首选，并保留仍可接受的其他选项；没有优先顺序也可以直接说明。", 45)

    candidates.sort(key=lambda item: item["priority"], reverse=True)
    eligible = [item for item in candidates if item["eligible"]]
    critical = [item for item in eligible if item["action"] == "resolve_conflict" or item["focus_field"] in _CORE_FIELDS]
    prioritization = [item for item in eligible if item["action"] == "prioritize"]
    selected, stop_reason, selection_reason = None, None, None
    if remaining == 0:
        stop_reason = "round_limit"
    elif critical:
        selected = critical[0]
        selection_reason = "unresolved_conflict" if selected["action"] == "resolve_conflict" else "essential_information"
    elif prioritization:
        selected = prioritization[0]
        selection_reason = "unranked_preferences"
    elif len(resolved) == len(states):
        stop_reason = "complete"
    elif not eligible:
        stop_reason = "no_eligible_question"
    elif metrics["information_completeness"] >= threshold:
        stop_reason = "sufficient_information"
    else:
        selected = eligible[0]
        selection_reason = "optional_information_below_threshold"
    return {
        "schema_version": "coaching-policy-v1", "policy_kind": "deterministic_rules",
        "action": selected["action"] if selected else "stop",
        "focus_field": selected["focus_field"] if selected else None,
        "question": selected["question"] if selected else None,
        "stop_reason": stop_reason, "selection_reason": selection_reason,
        "round": {"number": round_number, "max_rounds": max_rounds, "remaining_including_current": remaining},
        "metrics": metrics,
        "candidates": candidates,
        "action_rules": [dict(rule) for rule in _RULES],
        "unresolved_fields": [field for field in _PRIORITY if field not in resolved],
        "protected_fields": [field for field in _PRIORITY if states[field].status in _PROTECTED],
        "limitations": [
            "完整度和追问阈值是可解释的工程规则，不是已校准的事实概率或心理测量。",
            "阈值只调整是否继续询问，不修改证据、用户回答状态或事实可信度。",
            "本模块不推断情绪、不评估认知负荷，不包含PPO或其他策略训练。",
        ],
    }
