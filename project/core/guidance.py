"""Grounded local-model suggestions with deterministic state and question guards."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Protocol

from .intake import (
    FOLLOW_UP_QUESTIONS, add_evidence, answer_status, field_value,
    initialize_fields, record_answer, set_field,
)
from .schemas import (
    FieldState, GuidanceDecision, GuidanceObservation, GuidanceQuestion, PerceptionResult,
    TaskRequest, UserProfile,
)
from .coaching_policy import build_coaching_context


class GuidanceModel(Protocol):
    def analyze(self, context: dict) -> GuidanceDecision: ...
    def unload(self) -> None: ...


_PRIORITY = ("target_role", "skills", "time_budget", "education", "major", "interests", "constraints", "preference")
_TEMPLATES = {q.key: q.prompt for q in FOLLOW_UP_QUESTIONS}


def collect_evidence(profile: UserProfile, req: TaskRequest, results: list[PerceptionResult]) -> None:
    for origin, text in [("user_goal", req.user_goal), ("text_input", req.text_input)]:
        if origin == "text_input" and req.metadata.get("text_subject", "student") != "student":
            continue
        if text.strip():
            add_evidence(profile, origin, text, subject="student", verification="user_confirmed")
    for index, result in enumerate(results):
        if result.requires_confirmation or result.subject in {"job", "reference"}:
            continue
        if result.modality == "document" and result.raw_output.strip():
            add_evidence(profile, f"document:{index}:parsed_text", result.raw_output,
                         subject=result.subject, verification=result.verification, source_id=result.source_id)
        # Use quotations, not keyword matches or generated summaries as user claims.
        for number, evidence in enumerate(result.evidence):
            if evidence.quote.strip():
                add_evidence(profile, f"{result.modality}:{index}:evidence:{number}", evidence.quote,
                             subject=result.subject, verification=result.verification, source_id=result.source_id)


def _validate_observation(evidence: dict[str, str], observation: GuidanceObservation) -> None:
    if any(ref not in evidence for ref in observation.evidence_ids):
        raise ValueError("Unknown observation evidence")
    if not any(observation.value in evidence[ref] for ref in observation.evidence_ids):
        raise ValueError("Observation is not an evidence excerpt")


def _validate_question(evidence: dict[str, str], question: GuidanceQuestion | None) -> None:
    if question and any(ref not in evidence for ref in question.evidence_ids):
        raise ValueError("Unknown question evidence")


def merge_decision(profile: UserProfile, decision: GuidanceDecision) -> UserProfile:
    """Validate all references before applying anything; never overwrite a conflict."""
    evidence = {e.evidence_id: e.excerpt for e in profile.guidance.evidence}
    for observation in decision.observations:
        _validate_observation(evidence, observation)
        if any(item.subject in {"job", "reference"} and item.evidence_id in observation.evidence_ids
               for item in profile.guidance.evidence):
            raise ValueError("A reference requirement is not a student attribute")
    _validate_question(evidence, decision.question)
    updated = profile.model_copy(deep=True)
    for observation in decision.observations:
        field, value = observation.field, observation.value.strip()
        state = updated.guidance.fields[field]
        if set(observation.evidence_ids).issubset(state.resolved_evidence_ids):
            continue
        status = answer_status(field, value)
        if status == "missing":
            continue
        if status in {"unknown", "refused"}:
            if state.status == "missing":
                updated.guidance.fields[field] = FieldState(
                    status=status, evidence_ids=observation.evidence_ids,
                )
            continue
        normalized = updated.model_copy(deep=True)
        set_field(normalized, field, value if status == "known" else "")
        if status == "known":
            value = field_value(normalized, field)
        if state.status in {"unknown", "refused"}:
            continue
        old_value = field_value(updated, field)
        if state.status == status == "explicit_none":
            state.evidence_ids = list(dict.fromkeys([*state.evidence_ids, *observation.evidence_ids]))
            continue
        if state.status == "known" and status == "known" and old_value == value:
            state.evidence_ids = list(dict.fromkeys([*state.evidence_ids, *observation.evidence_ids]))
            continue
        if state.status == "known" and status == "known" and field in {"skills", "interests", "constraints"}:
            set_field(updated, field, f"{old_value}、{value}")
            state.values = [field_value(updated, field)]
            state.evidence_ids = list(dict.fromkeys([*state.evidence_ids, *observation.evidence_ids]))
            continue
        if state.status in {"known", "conflict", "explicit_none"}:
            state.status = "conflict"
            state.values = list(dict.fromkeys([*state.values, value]))
            state.evidence_ids = list(dict.fromkeys([*state.evidence_ids, *observation.evidence_ids]))
            set_field(updated, field, "")
        else:
            updated.guidance.fields[field] = FieldState(
                status=status, values=[value], evidence_ids=observation.evidence_ids,
            )
            set_field(updated, field, value if status == "known" else "")
    return updated


def question_key(profile: UserProfile, question: GuidanceQuestion) -> str:
    state = profile.guidance.fields[question.field]
    signature = [question.field, question.reason]
    if question.reason != "missing":
        signature.extend(sorted(state.evidence_ids))
        signature.extend(state.values)
    return hashlib.sha256(json.dumps(signature, ensure_ascii=False).encode()).hexdigest()[:20]


def eligible(profile: UserProfile, question: GuidanceQuestion) -> bool:
    state = profile.guidance.fields[question.field]
    if question_key(profile, question) in profile.guidance.asked_keys:
        return False
    if question.reason == "conflict":
        return state.status == "conflict"
    if question.reason == "missing":
        return state.status == "missing" and not any(a.field == question.field for a in profile.guidance.answers)
    # Clarification is allowed for a vague supplied goal, but never for unknown/refused answers.
    return (
        state.status == "known" and bool(question.evidence_ids)
        and bool(set(question.evidence_ids) & set(state.evidence_ids))
        and not any(a.field == question.field and a.origin == "interactive" for a in profile.guidance.answers)
    )


def choose_question(profile: UserProfile, suggested: GuidanceQuestion | None) -> GuidanceQuestion | None:
    for field in _PRIORITY:
        state = profile.guidance.fields[field]
        if state.status != "conflict":
            continue
        question = GuidanceQuestion(
            field=field, reason="conflict", evidence_ids=state.evidence_ids[:8],
            prompt=f"关于{_TEMPLATES[field]}已有信息不一致：{' / '.join(state.values)[:180]}。请直接给出当前准确信息。",
        )
        if eligible(profile, question):
            return question
    if suggested and eligible(profile, suggested):
        return suggested
    for field in _PRIORITY:
        question = GuidanceQuestion(field=field, reason="missing", prompt=_TEMPLATES[field])
        if eligible(profile, question):
            return question
    return None


@dataclass
class GuidanceStep:
    profile: UserProfile
    question: GuidanceQuestion | None = None
    question_id: str = ""


def advance_guidance(
    req: TaskRequest, profile: UserProfile, model: GuidanceModel, *,
    round_number: int, max_rounds: int, output_fn: Callable[[str], None] = print,
    unload: bool = True,
) -> GuidanceStep:
    """Generate at most one grounded question; never read input or re-perceive."""
    profile = profile.model_copy(deep=True)
    initialize_fields(profile)
    if req.skip_follow_up:
        profile.guidance.stop_reason = "skipped"
        return GuidanceStep(profile)
    if max_rounds == 0 or round_number > max_rounds:
        profile.guidance.stop_reason = "round_limit"
        return GuidanceStep(profile)
    policy_events = json.loads(req.metadata.get("coaching_trace", "[]"))
    try:
        try:
            policy_before = build_coaching_context(profile, round_number=round_number, max_rounds=max_rounds)
            # Use the same field names as the output contract. Internal names
            # such as education_stage previously leaked into model responses.
            model_profile = {field: field_value(profile, field) for field in _PRIORITY}
            model_profile["guidance"] = {"fields": {
                field: state.model_dump(exclude={"resolved_evidence_ids"})
                for field, state in profile.guidance.fields.items()
            }}
            decision = GuidanceDecision.model_validate(model.analyze({
                "user_goal": req.user_goal,
                "profile": model_profile,
                "evidence": [e.model_dump() for e in sorted(
                    profile.guidance.evidence,
                    key=lambda item: item.origin.startswith(("answers_json:", "stored_profile:", "interactive:")),
                )],
                "round_number": round_number,
                "coaching_policy": {
                    key: policy_before[key]
                    for key in ("policy_kind", "action", "focus_field", "metrics", "limitations")
                },
            }))
            evidence = {e.evidence_id: e.excerpt for e in profile.guidance.evidence}
            _validate_question(evidence, decision.question)
            valid_observations = []
            discarded = getattr(model, "last_discarded_observations", 0)
            rejected_observations = discarded if isinstance(discarded, int) else 0
            for observation in decision.observations:
                try:
                    _validate_observation(evidence, observation)
                except ValueError:
                    rejected_observations += 1
                else:
                    valid_observations.append(observation)
            if decision.observations and not valid_observations:
                raise ValueError("No grounded observations")
            filtered_decision = decision.model_copy(update={"observations": valid_observations})
            profile = merge_decision(profile, filtered_decision)
            profile.guidance.model_used = True
            if rejected_observations:
                output_fn("部分补充信息未通过结构或原文校验，已忽略；其余已确认信息已保留。")
        except Exception:
            # Do not echo exception strings: loaders may include private local paths.
            profile.guidance.stop_reason = "model_error"
            output_fn("本地需求引导不可用或输出未通过校验；已保留已知信息与未解决项。")
            return GuidanceStep(profile)
        policy = build_coaching_context(profile, round_number=round_number, max_rounds=max_rounds)
        goal_clarification = (
            decision.question is not None and decision.question.field == "target_role"
            and decision.question.reason == "clarify" and eligible(profile, decision.question)
            and not any(state.status == "conflict" for state in profile.guidance.fields.values())
        )
        # A grounded model can recognize a vague goal outside a finite rule
        # vocabulary. Retain that clarification before collecting background.
        if goal_clarification:
            policy = dict(policy, action="clarify_goal", focus_field="target_role",
                          question=decision.question.model_dump(), stop_reason=None,
                          selection_reason="grounded_model_clarification")
        policy_events.append({
            "round": round_number, "action": policy["action"], "focus_field": policy["focus_field"],
            "metrics": policy["metrics"], "stop_reason": policy["stop_reason"],
            "selection_reason": policy["selection_reason"],
        })
        req.metadata["coaching_trace"] = json.dumps(policy_events, ensure_ascii=False)
        if policy["stop_reason"]:
            profile.guidance.stop_reason = (
                "no_eligible_question" if policy["stop_reason"] == "complete" else policy["stop_reason"]
            )
            return GuidanceStep(profile)
        suggested = GuidanceQuestion.model_validate(policy["question"]) if policy["question"] else None
        if (decision.question is not None and suggested is not None
                and decision.question.field == suggested.field
                and decision.question.reason == suggested.reason
                and eligible(profile, decision.question)):
            suggested = decision.question
        question = suggested if suggested is not None and eligible(profile, suggested) else choose_question(profile, None)
        if question is None:
            profile.guidance.stop_reason = "no_eligible_question"
            return GuidanceStep(profile)
        key = question_key(profile, question)
        profile.guidance.stop_reason = "awaiting_answer"
        return GuidanceStep(profile, question, key)
    finally:
        if unload:
            model.unload()


def submit_guidance_answer(
    req: TaskRequest, profile: UserProfile, question: GuidanceQuestion, question_id: str,
    answer: str, *, round_number: int,
) -> UserProfile:
    """Apply an answer to its saved question without any model invocation."""
    profile = profile.model_copy(deep=True)
    if question_id != question_key(profile, question) or question_id in profile.guidance.asked_keys:
        raise ValueError("Question is stale")
    profile.guidance.asked_keys.append(question_id)
    if answer.strip() == "结束引导":
        profile.guidance.stop_reason = "user_stopped"
        return profile
    req.follow_up_answers[question.field] = answer
    record_answer(
        profile, question.field, answer, question_id=question_id, prompt=question.prompt,
        round_number=round_number, origin="interactive",
    )
    if profile.guidance.fields[question.field].status == "conflict":
        profile.guidance.asked_keys.append(question_key(profile, question))
    return profile


def run_guidance(
    req: TaskRequest, profile: UserProfile, results: list[PerceptionResult], model: GuidanceModel,
    *, max_rounds: int, input_fn: Callable[[str], str], output_fn: Callable[[str], None],
) -> UserProfile:
    profile = profile.model_copy(deep=True)
    initialize_fields(profile)
    collect_evidence(profile, req, results)
    profile.guidance.model_used = False
    # A CLI invocation starts a new trace; web steps preserve the saved trace.
    req.metadata.pop("coaching_trace", None)
    if req.skip_follow_up:
        profile.guidance.stop_reason = "skipped"
        return profile
    if max_rounds == 0:
        profile.guidance.stop_reason = "round_limit"
        return profile
    try:
        for round_number in range(1, max_rounds + 1):
            step = advance_guidance(
                req, profile, model, round_number=round_number, max_rounds=max_rounds,
                output_fn=output_fn, unload=False,
            )
            profile = step.profile
            if step.question is None:
                return profile
            try:
                answer = input_fn(f"{step.question.prompt}\n（可回答不知道、不想回答；输入‘结束引导’停止）\n> ")
            except (EOFError, KeyboardInterrupt):
                profile.guidance.stop_reason = "user_stopped"
                return profile
            profile = submit_guidance_answer(
                req, profile, step.question, step.question_id, answer, round_number=round_number,
            )
            if profile.guidance.stop_reason == "user_stopped":
                return profile
        profile.guidance.stop_reason = "round_limit"
        return profile
    finally:
        model.unload()
