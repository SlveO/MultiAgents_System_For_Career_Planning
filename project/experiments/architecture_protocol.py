"""Torch-free contracts for the controlled architecture experiment."""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jsonschema import Draft202012Validator

GROUP_MONOLITHIC = "monolithic"
GROUP_MODULAR_ONE_SHOT = "modular_one_shot"
GROUP_MODULAR_COLLABORATIVE = "modular_collaborative"
ALL_GROUPS: tuple[str, ...] = (
    GROUP_MONOLITHIC,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MODULAR_COLLABORATIVE,
)

ACTION_FINAL = "final"
ACTION_REQUEST_EVIDENCE = "request_evidence"
REASON_EVIDENCE_SUFFICIENT = "evidence_sufficient"
REASON_ROUND_CAP_REACHED = "round_cap_reached"
REASON_NO_RESOLVABLE_REQUEST = "no_resolvable_request"
REASON_MISSING_DECISIVE_EVIDENCE = "missing_decisive_evidence"
EVIDENCE_SUFFICIENT = "sufficient"
EVIDENCE_INSUFFICIENT = "insufficient"

EXPECTED_PROTOCOL_VERSION = "architecture-comparison-v3"
EXPECTED_SCHEMA_REFS = {
    "case": "dataset/research_case.schema.json",
    "evidence": "dataset/research_evidence.schema.json",
    "decision": "dataset/research_decision.schema.json",
    "plan": "dataset/research_plan.schema.json",
}
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_protocol(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the authoritative protocol and fail closed when it is unavailable."""
    root = (repo_root or default_repo_root()).resolve()
    path = root / "dataset" / "research_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("frozen research protocol is missing")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise ValueError("unsupported research protocol version")
    if protocol.get("schema_refs") != EXPECTED_SCHEMA_REFS:
        raise ValueError("research protocol schema references changed")
    return protocol


class ContractValidationError(ValueError):
    """A case or model payload violated one of the frozen JSON Schemas."""


class ResearchContracts:
    """Load and validate the four frozen Draft 2020-12 contracts once."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = (repo_root or default_repo_root()).resolve()
        self.protocol = load_protocol(self.repo_root)
        self._validators: Dict[str, Draft202012Validator] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._schema_paths: Dict[str, Path] = {}
        for kind, relative_path in self.protocol["schema_refs"].items():
            schema_path = self.repo_root / relative_path
            if not schema_path.is_file():
                raise FileNotFoundError(f"frozen {kind} schema is missing")
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            self._schemas[kind] = schema
            self._schema_paths[kind] = schema_path
            self._validators[kind] = Draft202012Validator(schema)

    def schema(self, kind: str) -> Dict[str, Any]:
        """Return a defensive copy of one canonical research Schema."""
        try:
            schema = self._schemas[kind]
        except KeyError as exc:
            raise ValueError(f"unknown research contract: {kind}") from exc
        return json.loads(json.dumps(schema))

    def schema_sha256(self, kind: str) -> str:
        """Hash the exact canonical Schema file bytes used for validation."""
        try:
            path = self._schema_paths[kind]
        except KeyError as exc:
            raise ValueError(f"unknown research contract: {kind}") from exc
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(self, kind: str, payload: Dict[str, Any]) -> None:
        try:
            validator = self._validators[kind]
        except KeyError as exc:
            raise ValueError(f"unknown research contract: {kind}") from exc
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
        if not errors:
            return
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ContractValidationError(
            f"{kind} payload invalid at {location}: {error.message}"
        )


def group_model_ids(protocol: Dict[str, Any], group: str) -> Dict[str, str]:
    entry = protocol["groups"][group]
    if group == GROUP_MONOLITHIC:
        return {"planner": entry["model_id"]}
    return {
        "perceiver": entry["perceiver_model_id"],
        "reasoner": entry["reasoner_model_id"],
    }


class PerceptionFailure(Exception):
    """Expected perceiver failure with a path-free public error code."""

    def __init__(self, code: str = "perception_failed") -> None:
        if not _SAFE_ERROR_CODE.fullmatch(code):
            raise ValueError("perception failure code must be a safe identifier")
        self.code = code
        super().__init__(code)


class BaseArchitectureAdapter(ABC):
    """Interface shared by offline fakes and L20 model adapters."""

    group: str = ""
    backend: str = ""

    @abstractmethod
    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        """Return one metadata record containing a Schema-valid ``plan``."""


class BoundedClarificationController:
    """Enforce request-plus-delta rounds and strict intermediate validation."""

    def __init__(
        self,
        round_cap: int,
        *,
        initial_perception: Callable[[Dict[str, Any]], Dict[str, Any]],
        decide: Callable[[Dict[str, Any], int], Dict[str, Any]],
        perceive_delta: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        finalize: Callable[[Dict[str, Any], int, Optional[str]], Dict[str, Any]],
        contracts: Optional[ResearchContracts] = None,
    ) -> None:
        if round_cap < 0:
            raise ValueError("round_cap must be non-negative")
        self.round_cap = round_cap
        self._initial_perception = initial_perception
        self._decide = decide
        self._perceive_delta = perceive_delta
        self._finalize = finalize
        self._contracts = contracts or ResearchContracts()

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        evidence = self._initial_perception(case)
        self._contracts.validate("evidence", evidence)
        rounds_used = 0
        error: Optional[str] = None

        while True:
            rounds_remaining = self.round_cap - rounds_used
            decision = self._decide(evidence, rounds_remaining)
            self._contracts.validate("decision", decision)
            if decision["action"] != ACTION_REQUEST_EVIDENCE:
                break
            if rounds_remaining <= 0:
                break
            try:
                delta = self._perceive_delta(case, decision)
            except PerceptionFailure as exc:
                error = exc.code
                break
            self._contracts.validate("evidence", delta)
            if delta["packet_type"] != "delta":
                raise ContractValidationError("clarification response must be a delta packet")
            if delta["request_id"] != decision["request_id"]:
                raise ContractValidationError("clarification request_id does not match decision")
            rounds_used += 1
            evidence = self._merge_evidence(evidence, delta)

        plan = self._finalize(evidence, rounds_used, error)
        self._contracts.validate("plan", plan)
        return {
            "rounds_used": rounds_used,
            "finalized": True,
            "error": error,
            "plan": plan,
        }

    @staticmethod
    def _merge_evidence(
        current: Dict[str, Any], delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        facts = list(current.get("facts", [])) + list(delta.get("facts", []))
        evidence_ids = [fact["evidence_id"] for fact in facts]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ContractValidationError("clarification produced a duplicate evidence_id")
        return {
            **current,
            "facts": facts,
            "missing_fields": list(delta.get("missing_fields", [])),
            "conflicts": list(current.get("conflicts", []))
            + list(delta.get("conflicts", [])),
        }
