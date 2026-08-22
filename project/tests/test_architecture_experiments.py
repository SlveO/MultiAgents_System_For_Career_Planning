from __future__ import annotations

import csv
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from project.experiments.architecture_protocol import (
    ACTION_REQUEST_EVIDENCE,
    BoundedClarificationController,
    ContractValidationError,
    GROUP_MODULAR_COLLABORATIVE,
    GROUP_MODULAR_ONE_SHOT,
    GROUP_MONOLITHIC,
    PerceptionFailure,
    ResearchContracts,
    load_protocol,
)
from project.experiments.run_architecture_experiments import (
    _build_plan,
    _evidence_packet,
    build_fake_adapters,
    load_cases,
    run_architecture_experiments,
)


class TestArchitectureExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = ResearchContracts()
        cls.cases = load_cases(contracts=cls.contracts)

    def test_imports_without_torch_in_fresh_interpreter(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import project.experiments.architecture_protocol; "
                "import project.experiments.run_architecture_experiments; "
                "assert 'torch' not in sys.modules"
            ),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_protocol_is_required_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "protocol is missing"):
                load_protocol(Path(tmp))

    def test_fake_run_emits_six_cases_for_all_three_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_architecture_experiments(output_dir=tmp, round_cap=2)
            json_rows = json.loads(
                (Path(tmp) / "results.json").read_text(encoding="utf-8")
            )
            with (Path(tmp) / "results.csv").open(encoding="utf-8-sig") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(rows, json_rows)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(csv_rows), 18)
        self.assertEqual(
            {row["group"] for row in rows},
            {GROUP_MONOLITHIC, GROUP_MODULAR_ONE_SHOT, GROUP_MODULAR_COLLABORATIVE},
        )
        self.assertTrue(all(row["backend"] == "offline-fake" for row in rows))

    def test_fake_plans_are_derived_from_each_case(self) -> None:
        expected = {
            case["case_id"]: (
                case["fixed_profile"]["target_role"],
                case["knowledge_snippets"][0]["knowledge_id"],
            )
            for case in self.cases
        }
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_architecture_experiments(output_dir=tmp, round_cap=3)

        for row in rows:
            target_role, knowledge_id = expected[row["case_id"]]
            self.assertEqual(row["plan"]["target_roles"], [target_role])
            self.assertEqual(row["plan"]["knowledge_ids_used"], [knowledge_id])

    def test_two_and_three_round_caps_have_distinct_deterministic_results(self) -> None:
        results = {}
        for cap in (2, 3):
            with tempfile.TemporaryDirectory() as tmp:
                rows = run_architecture_experiments(
                    output_dir=tmp,
                    round_cap=cap,
                    groups=[GROUP_MODULAR_COLLABORATIVE],
                )
            results[cap] = rows
            self.assertTrue(all(row["rounds_used"] == cap for row in rows))
            self.assertTrue(all(row["rounds_used"] <= row["round_cap"] for row in rows))

        self.assertTrue(
            all(row["plan"]["evidence_status"] == "insufficient" for row in results[2])
        )
        self.assertTrue(
            all(row["plan"]["evidence_status"] == "sufficient" for row in results[3])
        )

    def test_case_and_plan_contracts_fail_closed(self) -> None:
        invalid_case = copy.deepcopy(self.cases[0])
        invalid_case.pop("privacy_reviewed")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ContractValidationError):
                run_architecture_experiments(
                    output_dir=tmp,
                    cases=[invalid_case],
                    groups=[GROUP_MONOLITHIC],
                )

        invalid_plan = _build_plan(self.cases[0], [])
        invalid_plan["unexpected"] = True
        with self.assertRaises(ContractValidationError):
            self.contracts.validate("plan", invalid_plan)

    def test_asset_hash_mismatch_is_rejected(self) -> None:
        invalid_case = copy.deepcopy(self.cases[0])
        invalid_case["asset_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "cases.json"
            manifest.write_text(
                json.dumps({"cases": [invalid_case]}, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "asset_sha256 does not match"):
                load_cases(manifest, contracts=self.contracts)

    def test_perception_failure_uses_sanitized_code_and_finalizes(self) -> None:
        case = self.cases[0]

        def decide(evidence: dict, rounds_remaining: int) -> dict:
            return {
                "schema_version": "research-decision-v1",
                "action": ACTION_REQUEST_EVIDENCE,
                "reason_code": "missing_decisive_evidence",
                "request_id": "r-1",
                "target": "vision",
                "question": "请补充下一条证据。",
                "required_fields": ["ev-2"],
            }

        def fail(case_payload: dict, decision: dict) -> dict:
            raise PerceptionFailure("perceiver_unavailable")

        controller = BoundedClarificationController(
            round_cap=2,
            initial_perception=lambda active_case: _evidence_packet(
                active_case, [0], packet_type="initial", request_id=None
            ),
            decide=decide,
            perceive_delta=fail,
            finalize=lambda evidence, rounds, error: _build_plan(
                case, evidence["facts"], perception_error=error
            ),
            contracts=self.contracts,
        )
        result = controller.run(case)
        self.assertEqual(result["error"], "perceiver_unavailable")
        self.assertEqual(result["plan"]["evidence_status"], "insufficient")
        self.assertNotIn("\\", result["error"])

    def test_malformed_decision_is_rejected(self) -> None:
        case = self.cases[0]
        controller = BoundedClarificationController(
            round_cap=2,
            initial_perception=lambda active_case: _evidence_packet(
                active_case, [0], packet_type="initial", request_id=None
            ),
            decide=lambda evidence, remaining: {"action": "request_evidence"},
            perceive_delta=lambda active_case, decision: {},
            finalize=lambda evidence, rounds, error: _build_plan(case, evidence["facts"]),
            contracts=self.contracts,
        )
        with self.assertRaises(ContractValidationError):
            controller.run(case)

    def test_external_adapters_can_be_injected_without_gpu_imports(self) -> None:
        adapters = build_fake_adapters(self.contracts)
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_architecture_experiments(
                output_dir=tmp,
                round_cap=2,
                cases=[self.cases[0]],
                groups=[GROUP_MONOLITHIC],
                adapters=adapters,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group"], GROUP_MONOLITHIC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
