from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal["qa", "diagnosis", "planning", "review"]
ModalityType = Literal["text", "image", "document", "audio", "video"]


class UserConstraints(BaseModel):
    time_budget_hours_per_week: Optional[int] = None
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
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    current_stage: str = ""
    constraints: UserConstraints = Field(default_factory=UserConstraints)


class Milestone(BaseModel):
    period: Literal["30d", "90d", "180d"]
    objective: str
    deliverables: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)


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
    model_trace: List[str] = Field(default_factory=list)
    served_by: Literal["cloud_brain", "local_fallback"] = "local_fallback"
    retry_count: int = 0
    latency_ms: int = 0


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str = Field(..., min_length=1)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
