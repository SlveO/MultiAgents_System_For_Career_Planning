from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


IntentType = Literal["qa", "diagnosis", "planning", "review"]
ModalityType = Literal["text", "image", "document", "audio", "video"]
FeedbackChoice = Literal["过短", "合适", "过于详细"]


GuidanceField = Literal[
    "education", "major", "skills", "interests", "target_role",
    "time_budget", "preference", "constraints",
]
AnswerStatus = Literal["missing", "known", "unknown", "refused", "explicit_none", "conflict"]


class GuidanceEvidence(BaseModel):
    evidence_id: str
    origin: str
    excerpt: str


class FieldState(BaseModel):
    resolved_evidence_ids: List[str] = Field(default_factory=list)
    status: AnswerStatus = "missing"
    values: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    field: GuidanceField
    question_id: str
    prompt: str = ""
    raw_answer: str
    status: AnswerStatus
    round_number: int = 0
    origin: str = "answers_json"
    evidence_id: str


class GuidanceState(BaseModel):
    fields: Dict[GuidanceField, FieldState] = Field(default_factory=dict)
    evidence: List[GuidanceEvidence] = Field(default_factory=list)
    answers: List[AnswerRecord] = Field(default_factory=list)
    asked_keys: List[str] = Field(default_factory=list)
    stop_reason: str = "not_requested"
    model_used: bool = False


class GuidanceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: GuidanceField
    value: str = Field(min_length=1, max_length=500)
    evidence_ids: List[str] = Field(min_length=1, max_length=8)


class GuidanceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: GuidanceField
    reason: Literal["missing", "conflict", "clarify"]
    prompt: str = Field(min_length=1, max_length=300)
    evidence_ids: List[str] = Field(default_factory=list, max_length=8)


class GuidanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: List[GuidanceObservation] = Field(default_factory=list, max_length=24)
    question: Optional[GuidanceQuestion] = None


class UserConstraints(BaseModel):
    time_budget_hours_per_week: Optional[float] = Field(default=None, ge=0, le=168)
    financial_budget_cny: Optional[int] = None
    city: Optional[str] = None
    education_level: Optional[str] = None
    preferred_industries: List[str] = Field(default_factory=list)


class TaskRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    user_goal: str = Field(..., min_length=1)
    text_input: str = ""
    image_paths: List[str] = Field(default_factory=list)
    document_paths: List[str] = Field(default_factory=list)
    audio_paths: List[str] = Field(default_factory=list)
    video_paths: List[str] = Field(default_factory=list)
    brain_model: Optional[str] = None
    stream: bool = False
    debug_trace: bool = False
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    follow_up_answers: Dict[str, str] = Field(default_factory=dict)
    skip_follow_up: bool = False
    guidance: GuidanceState = Field(default_factory=GuidanceState)
    planner_mode: Literal["deepseek", "template"] = "deepseek"
    use_knowledge: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source: str
    quote: str


class PerceptionResult(BaseModel):
    modality: ModalityType
    summary: str
    facts: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_info: List[str] = Field(default_factory=list)
    raw_output: str = ""


class UserProfile(BaseModel):
    guidance: GuidanceState = Field(default_factory=GuidanceState)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    current_stage: str = ""
    education_stage: str = ""
    major: str = ""
    skills: List[str] = Field(default_factory=list)
    target_role: str = ""
    preference: str = ""
    main_constraints: List[str] = Field(default_factory=list)
    constraints: UserConstraints = Field(default_factory=UserConstraints)


class Milestone(BaseModel):
    period: Literal["30d", "90d", "180d"]
    objective: str
    deliverables: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)


DisplaySectionName = Literal[
    "advice", "targets", "gaps", "roadmap", "actions", "resources",
    "constraints", "risks", "questions",
]


class FeedbackDisplaySection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: DisplaySectionName
    item_ids: List[str] = Field(min_length=1, max_length=128)
    style: Literal["paragraph", "bullets"] = "bullets"


class FeedbackLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sections: List[FeedbackDisplaySection] = Field(min_length=1, max_length=9)


class OutputVersion(BaseModel):
    plan_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version: Optional[int] = Field(default=None, ge=1)
    display_text: str
    facts_sha256: str
    source_text_sha256: str
    item_ids: List[str] = Field(default_factory=list)
    created_at: str


class FeedbackAdaptationResult(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    session_id: str
    plan_id: str
    feedback: FeedbackChoice
    status: Literal["adapted", "unchanged", "failed"]
    output_version: OutputVersion
    source_version: int = 1
    reason: str = ""
    model_used: bool = False
    latency_ms: int = 0


class CareerPlanResponse(BaseModel):
    session_id: str
    intent: IntentType
    profile: UserProfile
    target_roles: List[str] = Field(default_factory=list)
    gap_analysis: List[str] = Field(default_factory=list)
    roadmap_30_90_180: List[Milestone] = Field(default_factory=list)
    learning_resources: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    user_facing_advice: str = ""
    perception_results: List[PerceptionResult] = Field(default_factory=list)
    knowledge_hits: List[str] = Field(default_factory=list)
    knowledge_hit_ids: List[str] = Field(default_factory=list)
    model_trace: List[str] = Field(default_factory=list)
    served_by: Literal["cloud_brain", "local_fallback"] = "local_fallback"
    retry_count: int = 0
    latency_ms: int = 0
    plan_id: str = ""
    output_version: Optional[OutputVersion] = None


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: FeedbackChoice
