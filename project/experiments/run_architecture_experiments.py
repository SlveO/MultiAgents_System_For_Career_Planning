"""Deterministic, torch-free runner for the architecture experiment plumbing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from project.experiments.architecture_protocol import (
    ACTION_FINAL,
    ACTION_REQUEST_EVIDENCE,
    ALL_GROUPS,
    BaseArchitectureAdapter,
    BoundedClarificationController,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_SUFFICIENT,
    GROUP_MODULAR_COLLABORATIVE,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MONOLITHIC,
    REASON_EVIDENCE_SUFFICIENT,
    REASON_MISSING_DECISIVE_EVIDENCE,
    REASON_ROUND_CAP_REACHED,
    ResearchContracts,
    group_model_ids,
)

DEFAULT_CASE_MANIFEST = "dataset/research_cases/pilot_cases.json"
OFFLINE_BACKEND = "offline-fake"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(
    manifest_path: str | Path = DEFAULT_CASE_MANIFEST,
    *,
    contracts: Optional[ResearchContracts] = None,
) -> List[Dict[str, Any]]:
    """Load, validate, and hash-check a versioned case manifest."""
    active_contracts = contracts or ResearchContracts()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = active_contracts.repo_root / path
    loaded = json.loads(path.read_text(encoding="utf-8"))
    cases = loaded.get("cases") if isinstance(loaded, dict) else loaded
    if not isinstance(cases, list) or not cases:
        raise ValueError("research case manifest must contain a non-empty cases list")

    _validate_cases(cases, active_contracts)
    return cases


def _validate_cases(
    cases: List[Dict[str, Any]], contracts: ResearchContracts
) -> None:
    seen_ids: set[str] = set()
    for case in cases:
        contracts.validate("case", case)
        case_id = case["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate research case_id: {case_id}")
        seen_ids.add(case_id)
        asset = (contracts.repo_root / case["asset_path"]).resolve()
        asset_root = (
            contracts.repo_root / "dataset" / "research_cases" / "assets"
        ).resolve()
        try:
            asset.relative_to(asset_root)
        except ValueError as exc:
            raise ValueError(f"case {case_id} asset escapes the versioned asset directory") from exc
        if not asset.is_file():
            raise FileNotFoundError(f"case {case_id} asset is missing")
        if _sha256(asset) != case["asset_sha256"]:
            raise ValueError(f"case {case_id} asset_sha256 does not match")


def _expected_fact(case: Dict[str, Any], expected_index: int) -> Dict[str, Any]:
    expected = case["expected_evidence"][expected_index]
    return {
        "evidence_id": f"f-{expected_index + 1}",
        "fact": expected["fact"],
        "supporting_detail": expected["career_implication"],
        "source_location": expected["location_hint"],
        "confidence": 0.9,
    }


def _evidence_packet(
    case: Dict[str, Any],
    expected_indexes: List[int],
    *,
    packet_type: str,
    request_id: Optional[str],
) -> Dict[str, Any]:
    included = set(expected_indexes)
    missing = [
        item["evidence_id"]
        for index, item in enumerate(case["expected_evidence"])
        if item["required_for_plan"] and index not in included
    ]
    return {
        "schema_version": "research-evidence-v1",
        "packet_type": packet_type,
        "request_id": request_id,
        "facts": [_expected_fact(case, index) for index in expected_indexes],
        "missing_fields": missing,
        "conflicts": [],
    }


def _evidence_used_from(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "evidence_ref": fact["evidence_id"],
            "fact": fact["fact"],
            "source_location": fact["source_location"],
            "supports": ["gap_analysis[0]", "next_actions[0]"],
        }
        for fact in facts
    ]


def _missing_required_evidence(
    case: Dict[str, Any], facts: List[Dict[str, Any]]
) -> List[str]:
    observed = {fact["fact"] for fact in facts}
    return [
        item["fact"]
        for item in case["expected_evidence"]
        if item["required_for_plan"] and item["fact"] not in observed
    ]


def _build_plan(
    case: Dict[str, Any],
    facts: List[Dict[str, Any]],
    *,
    perception_error: Optional[str] = None,
) -> Dict[str, Any]:
    profile = case["fixed_profile"]
    target_role = profile["target_role"]
    missing = _missing_required_evidence(case, facts)
    if perception_error and not missing:
        missing = ["感知失败后未能确认全部决定性媒体证据"]
    evidence_status = EVIDENCE_INSUFFICIENT if missing else EVIDENCE_SUFFICIENT
    constraints = list(profile.get("main_constraints", []))
    gap_seed = constraints[0] if constraints else "缺少与目标岗位直接对应的作品证据"
    risk_flags = [f"需验证限制：{item}" for item in constraints[:3]]
    if evidence_status == EVIDENCE_INSUFFICIENT:
        risk_flags.insert(0, "媒体证据不足，当前路线只能作为保守草案")
    if perception_error:
        risk_flags.insert(0, f"感知模块失败：{perception_error}")
    knowledge_ids = [item["knowledge_id"] for item in case["knowledge_snippets"]]
    return {
        "schema_version": "research-plan-v1",
        "target_roles": [target_role],
        "gap_analysis": [gap_seed, f"需把媒体中的岗位要求转化为{target_role}作品证据"],
        "roadmap_30_90_180": [
            {
                "period": "30d",
                "objective": f"完成{target_role}要求与个人能力盘点",
                "deliverables": ["岗位要求对照表"],
                "metrics": ["完成至少 10 条证据化差距记录"],
            },
            {
                "period": "90d",
                "objective": f"形成一个面向{target_role}的可展示项目",
                "deliverables": ["项目仓库或案例复盘"],
                "metrics": ["完成一次按岗位证据逐项验收"],
            },
            {
                "period": "180d",
                "objective": f"执行{target_role}投递与面试迭代",
                "deliverables": ["投递与面试复盘表"],
                "metrics": ["完成 30 次针对性投递并每周复盘"],
            },
        ],
        "learning_resources": [f"固定职业知识 {item}" for item in knowledge_ids],
        "next_actions": ["根据媒体证据更新一版岗位差距清单"],
        "risk_flags": risk_flags,
        "user_facing_advice": f"先补齐可核验的岗位证据，再推进{target_role}路线。",
        "confidence": 0.8 if evidence_status == EVIDENCE_SUFFICIENT else 0.45,
        "evidence_status": evidence_status,
        "missing_evidence": missing,
        "evidence_used": _evidence_used_from(facts),
        "knowledge_ids_used": knowledge_ids,
    }


def _fake_decision(
    evidence: Dict[str, Any], rounds_remaining: int, expected_count: int
) -> Dict[str, Any]:
    observed_count = len(evidence["facts"])
    if observed_count >= expected_count:
        return {
            "schema_version": "research-decision-v1",
            "action": ACTION_FINAL,
            "reason_code": REASON_EVIDENCE_SUFFICIENT,
        }
    if rounds_remaining <= 0:
        return {
            "schema_version": "research-decision-v1",
            "action": ACTION_FINAL,
            "reason_code": REASON_ROUND_CAP_REACHED,
        }
    return {
        "schema_version": "research-decision-v1",
        "action": ACTION_REQUEST_EVIDENCE,
        "reason_code": REASON_MISSING_DECISIVE_EVIDENCE,
        "request_id": f"r-{observed_count}",
        "target": "vision",
        "question": "请只定位下一条尚未确认的岗位证据。",
        "required_fields": [f"ev-{observed_count + 1}"],
    }


def _fake_delta_perception(
    case: Dict[str, Any], decision: Dict[str, Any]
) -> Dict[str, Any]:
    expected_index = int(decision["request_id"].split("-")[1])
    remaining_indexes = list(range(expected_index, len(case["expected_evidence"])))
    return _evidence_packet(
        case,
        [expected_index],
        packet_type="delta",
        request_id=decision["request_id"],
    ) | {
        "missing_fields": [
            case["expected_evidence"][index]["evidence_id"]
            for index in remaining_indexes[1:]
            if case["expected_evidence"][index]["required_for_plan"]
        ]
    }


def _make_run_record(
    case: Dict[str, Any],
    *,
    group: str,
    round_cap: int,
    rounds_used: int,
    error: Optional[str],
    plan: Dict[str, Any],
    model_ids: Dict[str, str],
    generation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "group": group,
        "modality": case["modality"],
        "backend": OFFLINE_BACKEND,
        "round_cap": round_cap,
        "rounds_used": rounds_used,
        "finalized": True,
        "error": error,
        "model_ids": model_ids,
        "model_revisions": {role: None for role in model_ids},
        "generation": generation,
        "latency_ms": 0.0,
        "peak_vram_mb": None,
        "retry_count": 0,
        "raw_invalid_output": None,
        "plan": plan,
    }


class _FakeAdapter(BaseArchitectureAdapter):
    backend = OFFLINE_BACKEND

    def __init__(
        self,
        model_ids: Dict[str, str],
        generation: Dict[str, Any],
        contracts: ResearchContracts,
    ) -> None:
        self.model_ids = model_ids
        self.generation = generation
        self.contracts = contracts

    def _record(
        self,
        case: Dict[str, Any],
        *,
        round_cap: int,
        rounds_used: int,
        error: Optional[str],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.contracts.validate("plan", plan)
        return _make_run_record(
            case,
            group=self.group,
            round_cap=round_cap,
            rounds_used=rounds_used,
            error=error,
            plan=plan,
            model_ids=self.model_ids,
            generation=self.generation,
        )


class FakeMonolithicAdapter(_FakeAdapter):
    group = GROUP_MONOLITHIC

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        facts = [_expected_fact(case, index) for index in range(len(case["expected_evidence"]))]
        return self._record(
            case,
            round_cap=0,
            rounds_used=0,
            error=None,
            plan=_build_plan(case, facts),
        )


class FakeModularOneShotAdapter(_FakeAdapter):
    group = GROUP_MODULAR_ONE_SHOT

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        facts = [_expected_fact(case, index) for index in range(len(case["expected_evidence"]))]
        return self._record(
            case,
            round_cap=0,
            rounds_used=0,
            error=None,
            plan=_build_plan(case, facts),
        )


class FakeModularCollaborativeAdapter(_FakeAdapter):
    group = GROUP_MODULAR_COLLABORATIVE

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        expected_count = len(case["expected_evidence"])
        controller = BoundedClarificationController(
            round_cap=round_cap,
            initial_perception=lambda active_case: _evidence_packet(
                active_case, [0], packet_type="initial", request_id=None
            ),
            decide=lambda evidence, remaining: _fake_decision(
                evidence, remaining, expected_count
            ),
            perceive_delta=_fake_delta_perception,
            finalize=lambda evidence, rounds, error: _build_plan(
                case, evidence["facts"], perception_error=error
            ),
            contracts=self.contracts,
        )
        result = controller.run(case)
        return self._record(
            case,
            round_cap=round_cap,
            rounds_used=result["rounds_used"],
            error=result["error"],
            plan=result["plan"],
        )


def build_fake_adapters(
    contracts: ResearchContracts,
) -> Dict[str, BaseArchitectureAdapter]:
    protocol = contracts.protocol
    generation = dict(protocol["generation"])
    return {
        GROUP_MONOLITHIC: FakeMonolithicAdapter(
            group_model_ids(protocol, GROUP_MONOLITHIC), generation, contracts
        ),
        GROUP_MODULAR_ONE_SHOT: FakeModularOneShotAdapter(
            group_model_ids(protocol, GROUP_MODULAR_ONE_SHOT), generation, contracts
        ),
        GROUP_MODULAR_COLLABORATIVE: FakeModularCollaborativeAdapter(
            group_model_ids(protocol, GROUP_MODULAR_COLLABORATIVE), generation, contracts
        ),
    }


def validate_run_record(
    row: Dict[str, Any], contracts: ResearchContracts
) -> None:
    required = {
        "case_id",
        "group",
        "modality",
        "backend",
        "round_cap",
        "rounds_used",
        "finalized",
        "error",
        "model_ids",
        "model_revisions",
        "generation",
        "latency_ms",
        "peak_vram_mb",
        "retry_count",
        "raw_invalid_output",
        "plan",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"run record is missing metadata fields: {', '.join(missing)}")
    if row["group"] not in ALL_GROUPS:
        raise ValueError("run record contains an unknown group")
    if row["rounds_used"] > row["round_cap"]:
        raise ValueError("run record exceeds its collaboration round cap")
    contracts.validate("plan", row["plan"])


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    plan = row["plan"]
    return {
        "case_id": row["case_id"],
        "group": row["group"],
        "modality": row["modality"],
        "backend": row["backend"],
        "round_cap": row["round_cap"],
        "rounds_used": row["rounds_used"],
        "error": row["error"] or "",
        "latency_ms": row["latency_ms"],
        "peak_vram_mb": row["peak_vram_mb"] if row["peak_vram_mb"] is not None else "",
        "evidence_status": plan["evidence_status"],
        "confidence": plan["confidence"],
        "target_roles": "|".join(plan["target_roles"]),
    }


def _write_results(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat_rows = [_flatten_row(row) for row in rows]
    with (output_dir / "results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def run_architecture_experiments(
    output_dir: str | Path = "data/experiments/architecture",
    *,
    round_cap: int = 2,
    cases: Optional[List[Dict[str, Any]]] = None,
    groups: Optional[List[str]] = None,
    repo_root: Optional[Path] = None,
    adapters: Optional[Mapping[str, BaseArchitectureAdapter]] = None,
) -> List[Dict[str, Any]]:
    """Run fake or injected adapters over Schema-valid, hash-checked cases."""
    contracts = ResearchContracts(repo_root)
    allowed_caps = contracts.protocol["groups"][GROUP_MODULAR_COLLABORATIVE][
        "pilot_round_caps"
    ]
    if round_cap not in allowed_caps:
        raise ValueError(f"round_cap must be one of {allowed_caps}")
    selected_cases = cases
    if selected_cases is None:
        selected_cases = load_cases(contracts=contracts)
    else:
        _validate_cases(selected_cases, contracts)
    selected_groups = groups or list(ALL_GROUPS)
    if not selected_groups or any(group not in ALL_GROUPS for group in selected_groups):
        raise ValueError("groups must contain known architecture groups")
    active_adapters = dict(adapters) if adapters is not None else build_fake_adapters(contracts)

    rows: List[Dict[str, Any]] = []
    for case in selected_cases:
        for group in selected_groups:
            try:
                adapter = active_adapters[group]
            except KeyError as exc:
                raise ValueError(f"missing adapter for group: {group}") from exc
            if adapter.group != group:
                raise ValueError(f"adapter group mismatch for {group}")
            row = adapter.run(case, round_cap=round_cap)
            validate_run_record(row, contracts)
            rows.append(row)

    _write_results(Path(output_dir), rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline three-group architecture plumbing check."
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments/architecture",
        help="Directory for results.json and results.csv.",
    )
    parser.add_argument(
        "--round-cap",
        type=int,
        default=2,
        choices=[2, 3],
        help="Pilot collaboration cap.",
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASE_MANIFEST,
        help="JSON manifest containing versioned research cases.",
    )
    args = parser.parse_args()
    contracts = ResearchContracts()
    cases = load_cases(args.cases, contracts=contracts)
    rows = run_architecture_experiments(
        args.output_dir,
        round_cap=args.round_cap,
        cases=cases,
        repo_root=contracts.repo_root,
    )
    print(f"Wrote {len(rows)} offline-fake rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
