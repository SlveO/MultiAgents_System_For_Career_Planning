"""Explicit subjects and confirmation states for extracted field claims.

Extraction, user confirmation and independent source verification are distinct.
Neither an upload nor an approval of a knowledge release verifies its facts.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Subject = Literal["student", "job", "reference", "unknown"]
Verification = Literal["unverified", "user_confirmed", "source_verified"]


class SourceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = ""
    field: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=5000)
    subject: Subject = "unknown"
    verification: Verification = "unverified"
    source_id: str = Field(default="", max_length=128)
    origin: str = Field(default="", max_length=300)
    quote: str = Field(default="", max_length=10000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    verification_reference: str = Field(default="", max_length=1000)
    verified_by: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def bind_claim(self) -> "SourceClaim":
        if self.verification == "source_verified" and not (
            self.verification_reference.strip() and self.verified_by.strip()
        ):
            raise ValueError("Source verification requires its reference and verifier")
        if not self.claim_id:
            self.claim_id = hashlib.sha256(
                f"{self.field}\n{self.value}\n{self.subject}\n{self.source_id}\n{self.origin}".encode()
            ).hexdigest()[:24]
        return self


def claim_from_excerpt(
    field: str, value: str, excerpt: str, *, subject: Subject = "unknown",
    source_id: str = "", origin: str = "", evidence_ids: list[str] | None = None,
) -> SourceClaim:
    """Create an unverified extraction, retaining an exact supporting excerpt."""
    if not value.strip() or value not in excerpt:
        raise ValueError("An extracted claim must be supported by its excerpt")
    return SourceClaim(
        field=field, value=value, quote=excerpt, subject=subject,
        source_id=source_id, origin=origin, evidence_ids=evidence_ids or [],
    )


def confirm_claim(claim: SourceClaim) -> SourceClaim:
    """A visitor can confirm a claim, without changing its subject or source trust."""
    updated = claim.model_dump()
    if claim.verification == "unverified":
        updated["verification"] = "user_confirmed"
    return SourceClaim.model_validate(updated)


def student_claims(
    claims: Iterable[SourceClaim | dict], *, require_confirmation: bool = True,
) -> list[SourceClaim]:
    """Select student claims only; job requirements never become capabilities."""
    result = []
    for raw in claims:
        claim = SourceClaim.model_validate(raw)
        if claim.subject != "student":
            continue
        if require_confirmation and claim.verification == "unverified":
            continue
        result.append(claim.model_copy(deep=True))
    return result
