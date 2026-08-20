"""Focused tests for the offline architecture-comparison runner.

Covers the four member-A acceptance criteria:
1. the experiment modules import without torch,
2. fake mode emits monolithic / modular_one_shot / modular_collaborative,
3. the 2/3 round cap is stable and a perception failure finalizes with an
   insufficient-evidence risk flag,
4. (non-import side: only new files are added; MVP entry and follow-ups untouched).
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from project.experiments.architecture_protocol import (
    ACTION_FINAL,
    ACTION_REQUEST_EVIDENCE,
    BoundedClarificationController,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_SUFFICIENT,
    GROUP_MODULAR_COLLABORATIVE,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MONOLITHIC,
    REASON_EVIDENCE_SUFFICIENT,
    REASON_MISSING_DECISIVE_EVIDENCE,
)
from project.experiments.run_architecture_experiments import (
    run_architecture_experiments,
)


def _initial_evidence(fact_count: int = 0) -> dict:
    facts = [
        {
            "evidence_id": f"f-{i}",
            "fact": f"事实 {i}",
            "supporting_detail": "详情",
            "source_location": "素材",
            "confidence": 0.9,
        }
        for i in range(1, fact_count + 1)
    ]
    return {"facts": facts, "missing_fields": [], "conflicts": []}


def _always_request(evidence: dict, rounds_remaining: int) -> dict:
    return {
        "schema_version": "research-decision-v1",
        "action": ACTION_REQUEST_EVIDENCE,
        "reason_code": REASON_MISSING_DECISIVE_EVIDENCE,
        "request_id": "r-1",
        "target": "vision",
        "question": "问题",
        "required_fields": ["x"],
    }


def _final_decision(evidence: dict, rounds_remaining: int) -> dict:
    return {
        "schema_version": "research-decision-v1",
        "action": ACTION_FINAL,
        "reason_code": REASON_EVIDENCE_SUFFICIENT,
    }


def _ok_delta(case: dict, decision: dict) -> dict:
    return {
        "facts": [
            {
                "evidence_id": "f-9",
                "fact": "新增",
                "supporting_detail": "d",
                "source_location": "s",
                "confidence": 0.8,
            }
        ],
        "missing_fields": [],
        "conflicts": [],
    }


def _raising_delta(case: dict, decision: dict) -> dict:
    raise RuntimeError("perceiver down")


def _finalize(evidence: dict, rounds_used: int, error: str | None) -> dict:
    facts = evidence.get("facts", [])
    insufficient = error is not None or not facts
    return {
        "schema_version": "research-plan-v1",
        "evidence_status": EVIDENCE_INSUFFICIENT if insufficient else EVIDENCE_SUFFICIENT,
        "missing_evidence": ["决定性证据缺失"] if insufficient else [],
        "evidence_used": [],
        "risk_flags": ["证据不足"] if insufficient else [],
    }


class TestArchitectureExperiments(unittest.TestCase):
    def test_imports_without_torch(self) -> None:
        # The modules are imported at the top of this file; if they required
        # torch the import would have failed before reaching here.
        self.assertNotIn("torch", sys.modules)

    def test_fake_run_emits_three_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_architecture_experiments(output_dir=Path(tmp))
            json_rows = json.loads(
                (Path(tmp) / "results.json").read_text(encoding="utf-8")
            )
            with (Path(tmp) / "results.csv").open(encoding="utf-8-sig") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(json_rows, rows)
        self.assertEqual(
            {row["group"] for row in rows},
            {GROUP_MONOLITHIC, GROUP_MODULAR_ONE_SHOT, GROUP_MODULAR_COLLABORATIVE},
        )
        # 2 demo cases x 3 groups = 6 records.
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(csv_rows), 6)
        for row in rows:
            plan = row["plan"]
            self.assertEqual(plan["schema_version"], "research-plan-v1")
            self.assertIn(plan["evidence_status"], (EVIDENCE_SUFFICIENT, EVIDENCE_INSUFFICIENT))
            self.assertEqual(len(plan["roadmap_30_90_180"]), 3)
            self.assertEqual(
                [m["period"] for m in plan["roadmap_30_90_180"]], ["30d", "90d", "180d"]
            )
            if plan["evidence_status"] == EVIDENCE_SUFFICIENT:
                self.assertEqual(plan["missing_evidence"], [])
                self.assertGreaterEqual(len(plan["evidence_used"]), 1)
            else:
                self.assertGreaterEqual(len(plan["missing_evidence"]), 1)

    def test_round_cap_is_enforced(self) -> None:
        controller = BoundedClarificationController(
            round_cap=3,
            initial_perception=lambda case: _initial_evidence(0),
            decide=_always_request,
            perceive_delta=_ok_delta,
            finalize=_finalize,
        )
        result = controller.run({"case_id": "image-001"})
        self.assertEqual(result["rounds_used"], 3)
        self.assertTrue(result["finalized"])
        self.assertIsNone(result["error"])

    def test_perception_failure_finalizes_with_insufficient_evidence(self) -> None:
        controller = BoundedClarificationController(
            round_cap=2,
            initial_perception=lambda case: _initial_evidence(0),
            decide=_always_request,
            perceive_delta=_raising_delta,
            finalize=_finalize,
        )
        result = controller.run({"case_id": "image-001"})
        self.assertEqual(result["rounds_used"], 0)
        self.assertEqual(result["error"], "perceiver down")
        self.assertEqual(result["plan"]["evidence_status"], EVIDENCE_INSUFFICIENT)
        self.assertIn("证据不足", result["plan"]["risk_flags"])
        self.assertTrue(result["plan"]["missing_evidence"])

    def test_collaborative_adapter_respects_round_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_architecture_experiments(
                output_dir=Path(tmp),
                round_cap=2,
                groups=[GROUP_MODULAR_COLLABORATIVE],
            )
        for row in rows:
            self.assertEqual(row["group"], GROUP_MODULAR_COLLABORATIVE)
            self.assertEqual(row["round_cap"], 2)
            self.assertLessEqual(row["rounds_used"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
