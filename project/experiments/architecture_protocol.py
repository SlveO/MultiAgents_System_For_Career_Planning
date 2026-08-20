"""Architecture-comparison experiment protocol (offline, torch-free).

This module consumes the frozen research contracts produced by the lead
(`dataset/research_protocol.json` and the four `dataset/research_*.schema.json`
files) and defines the adapter interface plus the bounded clarification
controller shared by every architecture group.

It intentionally does **not** import `torch`. Real model adapters live in a
separate module and run only on the L20 Ubuntu host; the offline fake adapters
in :mod:`project.experiments.run_architecture_experiments` subclass the same
interface and produce the same run-record shape.

Group names, decision action / reason codes, and evidence-status values are
frozen by the dataset contracts. Do not rename them here.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Frozen group keys from dataset/research_protocol.json -> groups.
GROUP_MONOLITHIC = "monolithic"
GROUP_MODULAR_ONE_SHOT = "modular_one_shot"
GROUP_MODULAR_COLLABORATIVE = "modular_collaborative"
ALL_GROUPS: tuple[str, ...] = (
    GROUP_MONOLITHIC,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MODULAR_COLLABORATIVE,
)

# Frozen decision actions from dataset/research_decision.schema.json.
ACTION_FINAL = "final"
ACTION_REQUEST_EVIDENCE = "request_evidence"

# Frozen final-decision reason codes from dataset/research_decision.schema.json.
REASON_EVIDENCE_SUFFICIENT = "evidence_sufficient"
REASON_ROUND_CAP_REACHED = "round_cap_reached"
REASON_NO_RESOLVABLE_REQUEST = "no_resolvable_request"
# Frozen reason code for the request_evidence action.
REASON_MISSING_DECISIVE_EVIDENCE = "missing_decisive_evidence"

# Frozen evidence-status values from dataset/research_plan.schema.json.
EVIDENCE_SUFFICIENT = "sufficient"
EVIDENCE_INSUFFICIENT = "insufficient"

# Frozen model identifiers from dataset/research_protocol.json -> groups. Kept as
# a fallback so the offline runner stays self-contained on a branch that predates
# the lead's frozen-contract commit; the authoritative source is read by
# :func:`load_protocol` when the file is present.
_FALLBACK_PROTOCOL: Dict[str, Any] = {
    "groups": {
        GROUP_MONOLITHIC: {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
        GROUP_MODULAR_ONE_SHOT: {
            "perceiver_model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "reasoner_model_id": "Qwen/Qwen3-4B-Instruct-2507",
        },
        GROUP_MODULAR_COLLABORATIVE: {
            "perceiver_model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "reasoner_model_id": "Qwen/Qwen3-4B-Instruct-2507",
        },
    }
}


def default_repo_root() -> Path:
    """Repository root, derived from this file's location (project/experiments/)."""
    return Path(__file__).resolve().parents[2]


def load_protocol(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the frozen research protocol, falling back to frozen model ids.

    The authoritative contract lives in ``dataset/research_protocol.json``. When
    the file is absent (e.g. on a member branch that predates the lead's commit)
    we return a minimal fallback containing only the frozen model ids, so the
    offline runner and its tests remain self-contained.
    """
    root = repo_root or default_repo_root()
    path = root / "dataset" / "research_protocol.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(json.dumps(_FALLBACK_PROTOCOL))


def group_model_ids(protocol: Dict[str, Any], group: str) -> Dict[str, str]:
    """Return the frozen model identifiers for a group, keyed by role."""
    entry = protocol["groups"][group]
    if group == GROUP_MONOLITHIC:
        return {"planner": entry["model_id"]}
    return {
        "perceiver": entry["perceiver_model_id"],
        "reasoner": entry["reasoner_model_id"],
    }


class PerceptionFailure(Exception):
    """A perceiver could not produce a delta evidence packet for one request."""


class BaseArchitectureAdapter(ABC):
    """Interface every architecture group implements.

    The offline fake adapters subclass this here; the real model adapters
    (torch) subclass it on the L20 host. Both produce the same run-record shape,
    with run metadata kept outside the ``plan`` payload (see the frozen
    ``run_metadata_outside_model_output`` policy).
    """

    group: str = ""

    @abstractmethod
    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        """Run one case and return a run record (metadata + ``plan``)."""


class BoundedClarificationController:
    """Enforce the frozen collaboration contract for the modular-collaborative group.

    One round is one ``request_evidence`` decision plus one perceiver ``delta``
    response; the decision call itself does not count (``decision_call_counts_as_round``
    is false). The loop always terminates: it stops when the reasoner finalizes,
    when the round cap is reached, or when a perceiver fails, and it finalizes
    with whatever evidence was gathered. Perceivers only ever append evidence and
    preserve conflicts (``clarification_response: delta_only``).
    """

    def __init__(
        self,
        round_cap: int,
        *,
        initial_perception: Callable[[Dict[str, Any]], Dict[str, Any]],
        decide: Callable[[Dict[str, Any], int], Dict[str, Any]],
        perceive_delta: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        finalize: Callable[[Dict[str, Any], int, Optional[str]], Dict[str, Any]],
    ) -> None:
        if round_cap < 0:
            raise ValueError("round_cap must be non-negative")
        self.round_cap = round_cap
        self._initial_perception = initial_perception
        self._decide = decide
        self._perceive_delta = perceive_delta
        self._finalize = finalize

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        evidence = self._initial_perception(case)
        rounds_used = 0
        error: Optional[str] = None

        while True:
            rounds_remaining = self.round_cap - rounds_used
            decision = self._decide(evidence, rounds_remaining)
            if decision.get("action") != ACTION_REQUEST_EVIDENCE:
                break
            if rounds_remaining <= 0:
                # The reasoner asked for evidence with no rounds left; the
                # controller enforces the cap and finalizes instead.
                break
            try:
                delta = self._perceive_delta(case, decision)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                break
            rounds_used += 1
            evidence = self._merge_evidence(evidence, delta)

        plan = self._finalize(evidence, rounds_used, error)
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
        # delta_only: append new facts, union missing fields, preserve conflicts.
        return {
            **current,
            "facts": list(current.get("facts", [])) + list(delta.get("facts", [])),
            "missing_fields": sorted(
                set(current.get("missing_fields", []))
                | set(delta.get("missing_fields", []))
            ),
            "conflicts": list(current.get("conflicts", []))
            + list(delta.get("conflicts", [])),
        }
