"""Grounded local-model suggestions with deterministic state and question guards."""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Protocol

from .intake import (
    FOLLOW_UP_QUESTIONS, add_evidence, answer_status, field_value,
    initialize_fields, record_answer, set_field,
)
from .schemas import (
    FieldState, GuidanceDecision, GuidanceObservation, GuidanceQuestion, PerceptionResult,
    TaskRequest, UserProfile,
)


class GuidanceModel(Protocol):
    def analyze(self, context: dict) -> GuidanceDecision: ...
    def unload(self) -> None: ...


_PRIORITY = ("target_role", "skills", "time_budget", "education", "major", "interests", "constraints", "preference")
_TEMPLATES = {q.key: q.prompt for q in FOLLOW_UP_QUESTIONS}


def collect_evidence(profile: UserProfile, req: TaskRequest, results: list[PerceptionResult]) -> None:
    for origin, text in [("user_goal", req.user_goal), ("text_input", req.text_input)]:
        if text.strip():
            add_evidence(profile, origin, text)
    for index, result in enumerate(results):
        if result.modality == "document" and result.raw_output.strip():
            add_evidence(profile, f"document:{index}:parsed_text", result.raw_output)
        # Use quotations, not keyword matches or generated summaries as user claims.
        for number, evidence in enumerate(result.evidence):
            if evidence.quote.strip():
                add_evidence(profile, f"{result.modality}:{index}:evidence:{number}", evidence.quote)


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


def run_guidance(
    req: TaskRequest, profile: UserProfile, results: list[PerceptionResult], model: GuidanceModel,
    *, max_rounds: int, input_fn: Callable[[str], str], output_fn: Callable[[str], None],
) -> UserProfile:
    profile = profile.model_copy(deep=True)
    initialize_fields(profile)
    collect_evidence(profile, req, results)
    profile.guidance.model_used = False
    if req.skip_follow_up:
        profile.guidance.stop_reason = "skipped"
        return profile
    if max_rounds == 0:
        profile.guidance.stop_reason = "round_limit"
        return profile
    try:
        for round_number in range(1, max_rounds + 1):
            try:
                model_profile = profile.model_dump(exclude={"guidance"})
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
                }))
                evidence = {e.evidence_id: e.excerpt for e in profile.guidance.evidence}
                _validate_question(evidence, decision.question)
                valid_observations = []
                rejected_observations = 0
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
                    output_fn("部分补充信息未能与原文对应，已忽略；其余已确认信息已保留。")
            except Exception:
                # Do not echo exception strings: loaders may include private local paths.
                profile.guidance.stop_reason = "model_error"
                output_fn("本地需求引导不可用或输出未通过校验；已保留已知信息与未解决项。")
                return profile
            question = choose_question(profile, decision.question)
            if question is None:
                profile.guidance.stop_reason = "no_eligible_question"
                return profile
            key = question_key(profile, question)
            try:
                answer = input_fn(f"{question.prompt}\n（可回答不知道、不想回答；输入‘结束引导’停止）\n> ")
            except (EOFError, KeyboardInterrupt):
                profile.guidance.stop_reason = "user_stopped"
                return profile
            profile.guidance.asked_keys.append(key)
            if answer.strip() == "结束引导":
                profile.guidance.stop_reason = "user_stopped"
                return profile
            req.follow_up_answers[question.field] = answer
            record_answer(
                profile, question.field, answer, question_id=key, prompt=question.prompt,
                round_number=round_number, origin="interactive",
            )
            # The conflict signature can change after a refusal; also suppress that signature.
            if profile.guidance.fields[question.field].status == "conflict":
                profile.guidance.asked_keys.append(question_key(profile, question))
        profile.guidance.stop_reason = "round_limit"
        return profile
    finally:
        model.unload()
