"""Offline architecture-comparison runner (torch-free).

Provides deterministic fake adapters for the three frozen architecture groups and
a runner that executes them over cases and writes ``results.json`` /
``results.csv``. Fake output only verifies the plumbing and result files; it is
not model-quality evidence (see the frozen ``formal_run_policy`` and the
evaluation rubric).

Run from the repository root::

    python -m project.experiments.run_architecture_experiments \
        --output-dir data/experiments/architecture --round-cap 2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    group_model_ids,
    load_protocol,
)

# Offline demo cases that structurally conform to dataset/research_case.schema.json.
# They are plumbing-only placeholders: no real media asset exists yet, so the
# asset_sha256 is a 64-zero placeholder and asset_path points at a not-yet-built
# file. Real cases arrive from the source-verification member later.
DEMO_CASES: List[Dict[str, Any]] = [
    {
        "schema_version": "research-case-v1",
        "case_id": "image-001",
        "sets": ["pilot"],
        "modality": "image",
        "asset_path": "dataset/research_cases/assets/image-001.png",
        "asset_sha256": "0" * 64,
        "source": {
            "source_type": "self_created",
            "source_url": "demo://placeholder",
            "license": "demo-placeholder",
        },
        "user_goal": "获得数据分析实习",
        "fixed_profile": {
            "education_stage": "本科大三",
            "major": "统计学",
            "skills": ["Python", "SQL"],
            "interests": ["数据分析"],
            "target_role": "数据分析师",
            "time_budget_hours_per_week": 10,
            "preference": "互联网行业",
            "main_constraints": ["项目经验不足"],
        },
        "knowledge_snippets": [
            {
                "knowledge_id": "career-001",
                "text": "（示例）数据分析师通常需要 SQL、统计和可视化能力。",
            }
        ],
        "expected_evidence": [
            {
                "evidence_id": "ev-1",
                "fact": "（示例）岗位要求 SQL",
                "location_hint": "素材顶部",
                "career_implication": "需强化 SQL",
                "required_for_plan": True,
            },
            {
                "evidence_id": "ev-2",
                "fact": "（示例）岗位要求可视化",
                "location_hint": "素材中部",
                "career_implication": "需强化可视化",
                "required_for_plan": True,
            },
            {
                "evidence_id": "ev-3",
                "fact": "（示例）岗位要求统计",
                "location_hint": "素材下部",
                "career_implication": "需强化统计",
                "required_for_plan": False,
            },
        ],
        "scoring_notes": {
            "acceptable_inferences": ["（示例）可推断需补项目作品集"],
            "forbidden_assumptions": ["（示例）不得假设已有大厂实习经历"],
        },
        "decisive_evidence_not_in_text": True,
        "privacy_reviewed": True,
    },
    {
        "schema_version": "research-case-v1",
        "case_id": "pdf-001",
        "sets": ["pilot"],
        "modality": "pdf_page",
        "page_number": 1,
        "asset_path": "dataset/research_cases/assets/pdf-001.pdf",
        "asset_sha256": "0" * 64,
        "source": {
            "source_type": "self_created",
            "source_url": "demo://placeholder",
            "license": "demo-placeholder",
        },
        "user_goal": "转行到产品运营",
        "fixed_profile": {
            "education_stage": "本科毕业",
            "major": "工商管理",
            "skills": ["Excel", "沟通"],
            "interests": ["用户运营"],
            "target_role": "产品运营",
            "time_budget_hours_per_week": 12,
            "preference": "消费行业",
            "main_constraints": ["无相关实习"],
        },
        "knowledge_snippets": [
            {
                "knowledge_id": "career-002",
                "text": "（示例）产品运营通常需要用户洞察、数据复盘和活动策划能力。",
            }
        ],
        "expected_evidence": [
            {
                "evidence_id": "ev-1",
                "fact": "（示例）岗位要求用户洞察",
                "location_hint": "第 1 页顶部",
                "career_implication": "需积累用户分析案例",
                "required_for_plan": True,
            },
            {
                "evidence_id": "ev-2",
                "fact": "（示例）岗位要求数据复盘",
                "location_hint": "第 1 页中部",
                "career_implication": "需强化数据复盘",
                "required_for_plan": True,
            },
            {
                "evidence_id": "ev-3",
                "fact": "（示例）岗位要求活动策划",
                "location_hint": "第 1 页下部",
                "career_implication": "需参与活动策划",
                "required_for_plan": False,
            },
        ],
        "scoring_notes": {
            "acceptable_inferences": ["（示例）可推断需准备转行作品集"],
            "forbidden_assumptions": ["（示例）不得假设已有运营实习"],
        },
        "decisive_evidence_not_in_text": True,
        "privacy_reviewed": True,
    },
]


def _build_plan(
    *,
    evidence_status: str,
    evidence_used: List[Dict[str, Any]],
    missing_evidence: List[str],
    risk_flags: List[str],
    knowledge_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a plan that structurally conforms to research_plan.schema.json."""
    return {
        "schema_version": "research-plan-v1",
        "target_roles": ["数据分析师"],
        "gap_analysis": ["缺少可量化的项目作品集", "岗位案例表达不足"],
        "roadmap_30_90_180": [
            {
                "period": "30d",
                "objective": "完成岗位与能力盘点",
                "deliverables": ["目标岗位 JD 清单"],
                "metrics": ["分析 10 份 JD"],
            },
            {
                "period": "90d",
                "objective": "完成一个数据分析项目",
                "deliverables": ["项目复盘文档"],
                "metrics": ["完成一次端到端分析"],
            },
            {
                "period": "180d",
                "objective": "完成投递与面试迭代",
                "deliverables": ["投递看板"],
                "metrics": ["完成 30 次有效投递"],
            },
        ],
        "learning_resources": ["SQLBolt", "Kaggle"],
        "next_actions": ["拆解两份目标岗位 JD"],
        "risk_flags": risk_flags,
        "user_facing_advice": "围绕目标岗位补齐项目证据，并按阶段复盘投递结果。",
        "confidence": 0.8,
        "evidence_status": evidence_status,
        "missing_evidence": missing_evidence,
        "evidence_used": evidence_used,
        "knowledge_ids_used": knowledge_ids or ["career-001"],
    }


def _evidence_used_from(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "evidence_ref": fact["evidence_id"],
            "fact": fact["fact"],
            "source_location": fact["source_location"],
            "supports": ["gap_analysis", "target_roles"],
        }
        for fact in facts
    ]


def _make_run_record(
    case: Dict[str, Any],
    *,
    group: str,
    round_cap: int,
    rounds_used: int,
    error: Optional[str],
    plan: Dict[str, Any],
    model_ids: Dict[str, str],
) -> Dict[str, Any]:
    # Run metadata stays outside `plan` (run_metadata_outside_model_output).
    return {
        "case_id": case["case_id"],
        "group": group,
        "modality": case.get("modality"),
        "round_cap": round_cap,
        "rounds_used": rounds_used,
        "finalized": True,
        "error": error,
        "model_ids": model_ids,
        "plan": plan,
    }


def _demo_initial_evidence() -> Dict[str, Any]:
    return {
        "schema_version": "research-evidence-v1",
        "packet_type": "initial",
        "request_id": None,
        "facts": [
            {
                "evidence_id": "f-1",
                "fact": "（示例）目标岗位要求掌握 SQL 与基础统计",
                "supporting_detail": "岗位 JD 第 1 条",
                "source_location": "素材顶部",
                "confidence": 0.9,
            }
        ],
        "missing_fields": [],
        "conflicts": [],
    }


def _fake_decision(
    evidence: Dict[str, Any], rounds_remaining: int
) -> Dict[str, Any]:
    if len(evidence.get("facts", [])) >= 2:
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
        "request_id": "r-1",
        "target": "vision",
        "question": "（示例）请确认目标岗位任职要求中是否包含数据可视化能力？",
        "required_fields": ["requirement", "location"],
    }


def _fake_delta_perception(
    case: Dict[str, Any], decision: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": "research-evidence-v1",
        "packet_type": "delta",
        "request_id": decision.get("request_id"),
        "facts": [
            {
                "evidence_id": "f-2",
                "fact": "（示例）岗位要求具备数据可视化能力",
                "supporting_detail": "岗位 JD 第 3 条",
                "source_location": "素材中部",
                "confidence": 0.85,
            }
        ],
        "missing_fields": [],
        "conflicts": [],
    }


def _fake_finalize(
    evidence: Dict[str, Any],
    rounds_used: int,
    perception_error: Optional[str] = None,
) -> Dict[str, Any]:
    facts = evidence.get("facts", [])
    if perception_error is not None or not facts:
        return _build_plan(
            evidence_status=EVIDENCE_INSUFFICIENT,
            evidence_used=_evidence_used_from(facts),
            missing_evidence=["（示例）决定性视觉证据缺失（感知失败或无证据）"],
            risk_flags=["证据不足：协作未获取到决定性媒体证据", "项目经验不足"],
        )
    return _build_plan(
        evidence_status=EVIDENCE_SUFFICIENT,
        evidence_used=_evidence_used_from(facts),
        missing_evidence=[],
        risk_flags=["项目经验不足，建议先补齐作品集"],
    )


class FakeMonolithicAdapter(BaseArchitectureAdapter):
    """Deterministic stand-in for the Qwen3-VL-8B monolithic planner."""

    group = GROUP_MONOLITHIC

    def __init__(self, model_ids: Dict[str, str]) -> None:
        self.model_ids = model_ids

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        facts = _demo_initial_evidence()["facts"]
        plan = _build_plan(
            evidence_status=EVIDENCE_SUFFICIENT,
            evidence_used=_evidence_used_from(facts),
            missing_evidence=[],
            risk_flags=["项目经验不足，建议先补齐作品集"],
        )
        return _make_run_record(
            case,
            group=self.group,
            round_cap=0,
            rounds_used=0,
            error=None,
            plan=plan,
            model_ids=self.model_ids,
        )


class FakeModularOneShotAdapter(BaseArchitectureAdapter):
    """Deterministic stand-in for the 2B perceiver -> 4B reasoner one-shot group."""

    group = GROUP_MODULAR_ONE_SHOT

    def __init__(self, model_ids: Dict[str, str]) -> None:
        self.model_ids = model_ids

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        facts = _demo_initial_evidence()["facts"]
        plan = _build_plan(
            evidence_status=EVIDENCE_SUFFICIENT,
            evidence_used=_evidence_used_from(facts),
            missing_evidence=[],
            risk_flags=["项目经验不足，建议先补齐作品集"],
        )
        return _make_run_record(
            case,
            group=self.group,
            round_cap=0,
            rounds_used=0,
            error=None,
            plan=plan,
            model_ids=self.model_ids,
        )


class FakeModularCollaborativeAdapter(BaseArchitectureAdapter):
    """Deterministic stand-in for the bounded collaborative clarification group."""

    group = GROUP_MODULAR_COLLABORATIVE

    def __init__(self, model_ids: Dict[str, str]) -> None:
        self.model_ids = model_ids

    def run(self, case: Dict[str, Any], round_cap: int = 2) -> Dict[str, Any]:
        controller = BoundedClarificationController(
            round_cap=round_cap,
            initial_perception=lambda c: _demo_initial_evidence(),
            decide=_fake_decision,
            perceive_delta=_fake_delta_perception,
            finalize=_fake_finalize,
        )
        result = controller.run(case)
        return _make_run_record(
            case,
            group=self.group,
            round_cap=round_cap,
            rounds_used=result["rounds_used"],
            error=result["error"],
            plan=result["plan"],
            model_ids=self.model_ids,
        )


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    plan = row.get("plan", {})
    return {
        "case_id": row["case_id"],
        "group": row["group"],
        "modality": row["modality"],
        "round_cap": row["round_cap"],
        "rounds_used": row["rounds_used"],
        "finalized": row["finalized"],
        "error": row["error"] or "",
        "evidence_status": plan.get("evidence_status", ""),
        "confidence": plan.get("confidence", ""),
        "target_roles": "|".join(plan.get("target_roles", [])),
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
) -> List[Dict[str, Any]]:
    """Run the offline three-group comparison and write results.json/csv.

    Returns the run records; one record per ``(case, group)`` combination. The
    collaborative group runs through :class:`BoundedClarificationController` with
    the given ``round_cap`` (2 or 3); the other two groups are single-shot.
    """
    protocol = load_protocol(repo_root)
    selected_cases = cases if cases is not None else DEMO_CASES
    selected_groups = groups if groups is not None else list(ALL_GROUPS)

    adapters: Dict[str, BaseArchitectureAdapter] = {
        GROUP_MONOLITHIC: FakeMonolithicAdapter(
            group_model_ids(protocol, GROUP_MONOLITHIC)
        ),
        GROUP_MODULAR_ONE_SHOT: FakeModularOneShotAdapter(
            group_model_ids(protocol, GROUP_MODULAR_ONE_SHOT)
        ),
        GROUP_MODULAR_COLLABORATIVE: FakeModularCollaborativeAdapter(
            group_model_ids(protocol, GROUP_MODULAR_COLLABORATIVE)
        ),
    }

    rows: List[Dict[str, Any]] = []
    for case in selected_cases:
        for group in selected_groups:
            rows.append(adapters[group].run(case, round_cap=round_cap))

    _write_results(Path(output_dir), rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline three-group architecture comparison (fake adapters)."
    )
    parser.add_argument(
        "--output-dir",
        default="data/experiments/architecture",
        help="Directory for results.json / results.csv.",
    )
    parser.add_argument(
        "--round-cap",
        type=int,
        default=2,
        choices=[2, 3],
        help="Collaboration round cap for the pilot (2 or 3).",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional path to a JSON case manifest; defaults to built-in demo cases.",
    )
    args = parser.parse_args()

    cases: Optional[List[Dict[str, Any]]] = None
    if args.cases:
        loaded = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        cases = loaded.get("cases", []) if isinstance(loaded, dict) else loaded

    rows = run_architecture_experiments(
        args.output_dir, round_cap=args.round_cap, cases=cases
    )
    print(f"Wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
