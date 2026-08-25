from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from project.experiments.architecture_protocol import ResearchContracts
from project.experiments.local_models import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODELS_DIR,
    FROZEN_MODELS,
    L20ModularCollaborativeAdapter,
    L20ModularOneShotAdapter,
    LocalModelClient,
    ManualPreflightError,
    ModelNotDownloadedError,
    PromptCatalog,
    _smoke_outcome_fields,
    get_entry,
    local_model_dir,
    validate_manual_preflight,
)
from project.experiments.run_architecture_experiments import _build_plan, _evidence_packet, load_cases
from project.experiments.structured_output import StructuredOutputError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _preflight_payload(checked_at: datetime, *, status: str = "PASS") -> dict:
    return {
        "schema_version": "manual-l20-preflight-v1",
        "checked_at_utc": checked_at.isoformat().replace("+00:00", "Z"),
        "status": status,
        "human_confirmed_no_process_conflict": True,
        "selected_gpu": {
            "index": 0,
            "name": "NVIDIA L20",
            "driver_version": "test",
            "memory_total_mib": 46068,
            "memory_used_mib": 0,
            "memory_free_mib": 46068,
        },
    }


class FakeClient:
    def __init__(self, payloads: list[dict], model_id: str) -> None:
        self.payloads = list(payloads)
        self.entry = get_entry(model_id)
        self.media_paths = []
        self.constraints = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        media_path=None,
        constraint=None,
    ) -> dict:
        self.media_paths.append(media_path)
        self.constraints.append(constraint)
        payload = self.payloads.pop(0)
        return {
            "output_text": json.dumps(payload, ensure_ascii=False),
            "latency_ms": 1.0,
            "peak_vram_mb": 10,
            "device": "cuda:0",
            "visual_tokens": 20 if media_path else 0,
        }


class ConstraintFailClient(FakeClient):
    def generate(self, *args, **kwargs) -> dict:
        self.constraints.append(kwargs.get("constraint"))
        raise StructuredOutputError("test constraint failure")


class TestFrozenRegistry(unittest.TestCase):
    def test_registry_and_canonical_model_root(self) -> None:
        self.assertEqual(len(FROZEN_MODELS), 3)
        self.assertEqual({entry["role"] for entry in FROZEN_MODELS}, {"monolithic", "perceiver", "reasoner"})
        self.assertEqual(DEFAULT_MODELS_DIR, REPO_ROOT.parents[1] / "models")
        for entry in FROZEN_MODELS:
            self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
            self.assertEqual(local_model_dir(entry["model_id"]).parent, DEFAULT_MODELS_DIR)

    def test_prompt_hashes_match_protocol(self) -> None:
        catalog = PromptCatalog(ResearchContracts())
        self.assertEqual(catalog.version, "research-prompts-v3")
        self.assertEqual(len(catalog.hashes), 5)
        case = load_cases(contracts=ResearchContracts())[0]
        self.assertIn("output_schema=", catalog.perception_initial(case)[1])
        self.assertIn(
            "research-plan-v1", catalog.planning(case, None, raw_media=True)[1]
        )
        self.assertIn(
            "research-decision-v1", catalog.decision(case, {"facts": []}, 0)[1]
        )


class TestManualPreflight(unittest.TestCase):
    def test_accepts_fresh_human_pass(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight.json"
            path.write_text(json.dumps(_preflight_payload(now)), encoding="utf-8")
            record = validate_manual_preflight(path, now=now)
        self.assertEqual(record["selected_gpu"]["name"], "NVIDIA L20")

    def test_rejects_stale_or_failed_record(self) -> None:
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight.json"
            path.write_text(json.dumps(_preflight_payload(now - timedelta(minutes=11))), encoding="utf-8")
            with self.assertRaisesRegex(ManualPreflightError, "stale"):
                validate_manual_preflight(path, now=now)
            path.write_text(json.dumps(_preflight_payload(now, status="FAIL")), encoding="utf-8")
            with self.assertRaisesRegex(ManualPreflightError, "did not pass"):
                validate_manual_preflight(path, now=now)


class TestClientWithoutGPU(unittest.TestCase):
    def test_module_imports_without_heavy_dependencies(self) -> None:
        code = (
            "import sys; sys.modules['torch']=None; sys.modules['transformers']=None; "
            "sys.modules['fitz']=None; import project.experiments.local_models; "
            "assert 'torch' in sys.modules and sys.modules['torch'] is None"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_weights_fails_before_gpu_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = LocalModelClient(FROZEN_MODELS[2]["model_id"], models_dir=Path(tmp))
            with self.assertRaises(ModelNotDownloadedError):
                client.load()

    def test_deterministic_defaults(self) -> None:
        client = LocalModelClient(FROZEN_MODELS[2]["model_id"])
        self.assertEqual(client.seed, 42)
        self.assertEqual(client.max_new_tokens, DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(client.max_new_tokens, 2048)

    def test_smoke_outcome_metrics_separate_pipeline_and_content(self) -> None:
        generated = {"output_text": "{}"}
        valid = _smoke_outcome_fields(None, generated)
        content_failure = _smoke_outcome_fields(
            "smoke_grounding_failed", generated
        )
        constraint_failure = _smoke_outcome_fields(
            "constraint_failed", None
        )
        self.assertTrue(valid["pipeline_success"])
        self.assertTrue(content_failure["pipeline_success"])
        self.assertIsNone(content_failure["semantic_content_valid"])
        self.assertFalse(constraint_failure["pipeline_success"])
        self.assertIsNone(constraint_failure["raw_json_valid"])


class TestRealAdapterBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = ResearchContracts()
        cls.case = next(case for case in load_cases(contracts=cls.contracts) if case["modality"] == "image")

    def test_one_shot_never_sends_media_to_reasoner(self) -> None:
        evidence = _evidence_packet(
            self.case,
            list(range(len(self.case["expected_evidence"]))),
            packet_type="initial",
            request_id=None,
        )
        plan = _build_plan(self.case, evidence["facts"])
        perceiver = FakeClient([evidence], FROZEN_MODELS[1]["model_id"])
        reasoner = FakeClient([plan], FROZEN_MODELS[2]["model_id"])
        adapter = L20ModularOneShotAdapter(
            contracts=self.contracts,
            clients={"perceiver": perceiver, "reasoner": reasoner},
        )
        row = adapter.run(self.case)
        self.assertIsNotNone(perceiver.media_paths[0])
        self.assertIsNone(reasoner.media_paths[0])
        self.assertEqual(perceiver.constraints[0].kind, "evidence")
        self.assertEqual(reasoner.constraints[0].kind, "plan")
        self.assertEqual(row["backend"], "l20")
        self.assertEqual(row["retry_count"], 0)
        self.assertEqual(row["error"], None)
        self.assertTrue(row["pipeline_success"])
        self.assertIsNone(row["semantic_content_valid"])
        self.assertIsNone(row["evidence_faithfulness"])
        self.assertIsNone(row["task_success"])

    def test_collaboration_rejects_repeated_request_id(self) -> None:
        initial = _evidence_packet(
            self.case, [0], packet_type="initial", request_id=None
        )
        delta = _evidence_packet(
            self.case, [1], packet_type="delta", request_id="r-1"
        )
        repeated_decision = {
            "schema_version": "research-decision-v1",
            "action": "request_evidence",
            "reason_code": "missing_decisive_evidence",
            "request_id": "r-1",
            "target": "vision",
            "question": "请补充一条证据。",
            "required_fields": ["ev-2"],
        }
        perceiver = FakeClient(
            [initial, delta], FROZEN_MODELS[1]["model_id"]
        )
        reasoner = FakeClient(
            [repeated_decision, repeated_decision],
            FROZEN_MODELS[2]["model_id"],
        )
        adapter = L20ModularCollaborativeAdapter(
            contracts=self.contracts,
            clients={"perceiver": perceiver, "reasoner": reasoner},
        )
        row = adapter.run(self.case, round_cap=2)
        self.assertEqual(row["rounds_used"], 1)
        self.assertEqual(row["error"], "schema_invalid")
        self.assertEqual(row["retry_count"], 0)

    def test_constraint_failure_is_explicit_and_never_falls_back(self) -> None:
        perceiver = ConstraintFailClient([], FROZEN_MODELS[1]["model_id"])
        reasoner = FakeClient([], FROZEN_MODELS[2]["model_id"])
        adapter = L20ModularOneShotAdapter(
            contracts=self.contracts,
            clients={"perceiver": perceiver, "reasoner": reasoner},
        )
        row = adapter.run(self.case)
        self.assertEqual(row["error"], "constraint_failed")
        self.assertFalse(row["pipeline_success"])
        self.assertFalse(row["fallback_used"])
        self.assertFalse(row["repair_used"])
        self.assertEqual(len(reasoner.constraints), 0)

    def test_canonical_schema_validation_still_runs_after_generation(self) -> None:
        evidence = _evidence_packet(
            self.case,
            list(range(len(self.case["expected_evidence"]))),
            packet_type="initial",
            request_id=None,
        )
        plan = _build_plan(self.case, evidence["facts"])
        plan["confidence"] = 2.0
        perceiver = FakeClient([evidence], FROZEN_MODELS[1]["model_id"])
        reasoner = FakeClient([plan], FROZEN_MODELS[2]["model_id"])
        adapter = L20ModularOneShotAdapter(
            contracts=self.contracts,
            clients={"perceiver": perceiver, "reasoner": reasoner},
        )

        row = adapter.run(self.case)

        self.assertEqual(row["error"], "schema_invalid")
        self.assertFalse(row["pipeline_success"])
        self.assertIsNotNone(row["raw_invalid_output"])
        self.assertEqual(len(reasoner.constraints), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
