from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from project.core.output_adaptation import adapt_output, facts_hash, original_version, text_hash
from project.core.privacy import redact_data, redact_text, session_key
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import CareerPlanResponse, EvidenceItem, PerceptionResult, TaskRequest
from project.core.session_memory import SessionMemory
from project.orchestrator import CareerOrchestrator
from project.tests.test_output_adaptation import AdaptableBrain, FakeFeedbackModel, sample_response


def two_rounds():
    response = sample_response()
    expanded = adapt_output(response, "过短", "round-1", FakeFeedbackModel())
    compressed = adapt_output(
        response, "过于详细", "round-2", FakeFeedbackModel(), baseline=expanded.output_version,
    )
    return response, expanded, compressed


class TestFeedbackRoundAdaptation(unittest.TestCase):
    def test_third_version_compares_latest_display_and_preserves_canonical_plan(self):
        response = sample_response()
        original = response.model_dump()
        expanded = adapt_output(response, "过短", "round-1", FakeFeedbackModel())
        self.assertEqual(expanded.status, "adapted")
        model = FakeFeedbackModel()
        compressed = adapt_output(
            response, "过于详细", "round-2", model, baseline=expanded.output_version,
        )
        self.assertEqual(compressed.status, "adapted")
        self.assertEqual(compressed.source_version, 2)
        self.assertEqual(compressed.output_version.version, 3)
        self.assertEqual(compressed.output_version.parent_version, 2)
        self.assertEqual(compressed.output_version.display_text, expanded.output_version.display_text)
        self.assertTrue(compressed.output_version.reading_view.excerpt)
        self.assertEqual(response.model_dump(), original)
        self.assertFalse(model.calls)

    def test_suitable_uses_latest_display_without_model_or_new_version(self):
        response, expanded, _ = two_rounds()
        model = FakeFeedbackModel(RuntimeError("must not load"))
        result = adapt_output(response, "合适", "suitable", model, baseline=expanded.output_version)
        self.assertEqual(result.status, "unchanged")
        self.assertEqual(result.source_version, 2)
        self.assertEqual(result.output_version, expanded.output_version)
        self.assertFalse(model.calls)

    def test_foreign_or_changed_baseline_cannot_be_adapted(self):
        response, expanded, _ = two_rounds()
        for change in ({"plan_id": "another-plan"}, {"facts_sha256": "changed"},
                       {"source_text_sha256": "changed"}):
            with self.subTest(change=change):
                model = FakeFeedbackModel()
                baseline = expanded.output_version.model_copy(update=change)
                result = adapt_output(response, "过于详细", "bad", model, baseline=baseline)
                self.assertEqual((result.status, result.reason), ("failed", "source_changed"))
                self.assertFalse(model.calls)
        response.target_roles.append("未经确认岗位")
        result = adapt_output(response, "合适", "changed-plan", FakeFeedbackModel(),
                              baseline=expanded.output_version)
        self.assertEqual(result.reason, "source_changed")

    def test_privacy_safe_hash_survives_persistence_and_evidence_rehydration(self):
        response = sample_response()
        response.user_facing_advice = "联系 demo@example.com 后盘点能力。"
        response.profile.main_constraints.append("联系号码13812345678")
        response.perception_results.append(PerceptionResult(
            modality="document", summary="匿名材料", facts=["联系 demo@example.com"],
            evidence=[EvidenceItem(source="/private/student.docx", quote="13812345678")],
            raw_output="原始私人材料",
        ))
        response.output_version = original_version(response)
        payload = redact_data(response.model_dump())
        for perception in payload["perception_results"]:
            for evidence in perception["evidence"]:
                evidence.setdefault("source", "")
                evidence.setdefault("quote", "")
        restored = CareerPlanResponse.model_validate(payload)
        self.assertEqual(facts_hash(response), facts_hash(restored))
        self.assertEqual(response.output_version.source_text_sha256,
                         text_hash(restored.user_facing_advice))
        self.assertEqual(response.output_version.display_text, redact_text(response.user_facing_advice))
        expanded = adapt_output(restored, "过短", "private-round-1", FakeFeedbackModel())
        self.assertEqual(expanded.status, "adapted")
        self.assertNotIn("demo@example.com", expanded.model_dump_json())
        self.assertNotIn("13812345678", expanded.model_dump_json())
        self.assertIn("[REDACTED_PHONE]", expanded.output_version.display_text)
        # A legacy hash of private source text is rejected, never silently rewritten.
        legacy = response.output_version.model_copy(update={
            "source_text_sha256": text_hash(response.user_facing_advice),
        })
        rejected = adapt_output(restored, "过短", "legacy", FakeFeedbackModel(), baseline=legacy)
        self.assertEqual((rejected.status, rejected.reason), ("failed", "source_changed"))


class TestFeedbackRoundPersistence(unittest.TestCase):
    @staticmethod
    def memory(root, backend):
        memory = SessionMemory(str(root / "sessions.db"))
        memory.backend = backend
        return memory

    @staticmethod
    def seed(memory, response):
        request = TaskRequest(session_id=response.session_id, user_goal="匿名职业规划")
        memory.append_interaction(response.session_id, request.model_dump(), response.model_dump())
        memory.save_output_version(response.session_id, response.output_version.model_dump())
        return request

    def test_sqlite_and_json_restore_full_version_chain_and_canonical_plan(self):
        for backend in ("sqlite", "json"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                response, expanded, compressed = two_rounds()
                memory = self.memory(root, backend)
                request = self.seed(memory, response)
                self.assertTrue(memory.save_feedback_result(expanded.model_dump()))
                restarted = self.memory(root, backend)
                stored = restarted.get_plan_interaction("demo", response.plan_id)
                self.assertEqual(stored["request"], request.model_dump())
                self.assertEqual(stored["response"], response.model_dump())
                self.assertTrue(restarted.save_feedback_result(compressed.model_dump()))
                self.assertFalse(restarted.save_feedback_result(compressed.model_dump()))
                versions = restarted.get_output_versions("demo", response.plan_id)
                self.assertEqual([item["version"] for item in versions], [1, 2, 3])
                self.assertEqual([item["parent_version"] for item in versions], [None, 1, 2])
                self.assertEqual(restarted.get_latest_output_version("demo"), compressed.output_version.model_dump())
                self.assertIsNone(restarted.get_plan_interaction("another-session", response.plan_id))

    def test_plan_lookup_is_not_limited_to_recent_twenty_interactions(self):
        for backend in ("sqlite", "json"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                memory = self.memory(Path(tmp), backend)
                response = sample_response()
                self.seed(memory, response)
                for index in range(25):
                    memory.append_interaction("demo", {}, {"plan_id": f"later-{index}"})
                self.assertEqual(len(memory.get_session_history("demo")), 20)
                self.assertEqual(memory.get_plan_interaction("demo", response.plan_id)["response"],
                                 response.model_dump())
                self.assertIsNone(memory.get_plan_interaction("demo", "absent"))

    def test_stale_parent_changed_hash_and_foreign_session_are_rejected_atomically(self):
        for backend in ("sqlite", "json"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                memory = self.memory(Path(tmp), backend)
                response, expanded, compressed = two_rounds()
                self.seed(memory, response)
                memory.save_feedback_result(expanded.model_dump())
                bad_results = []
                stale = expanded.model_dump()
                stale["request_id"] = "stale"
                bad_results.append(stale)
                wrong_hash = compressed.model_dump()
                wrong_hash["request_id"] = "hash"
                wrong_hash["output_version"]["facts_sha256"] = "different-facts"
                bad_results.append(wrong_hash)
                wrong_session = compressed.model_dump()
                wrong_session["request_id"] = "session"
                wrong_session["session_id"] = "another-session"
                bad_results.append(wrong_session)
                for bad in bad_results:
                    with self.subTest(request_id=bad["request_id"]):
                        with self.assertRaisesRegex(ValueError, "latest version"):
                            memory.save_feedback_result(bad)
                        self.assertIsNone(memory.get_feedback_result(bad["request_id"]))
                self.assertEqual(len(memory.get_output_versions("demo", response.plan_id)), 2)


class TestFeedbackRoundLogging(unittest.TestCase):
    def test_full_third_version_chain_is_ordered_immutable_and_idempotent(self):
        response, expanded, compressed = two_rounds()
        versions = [response.output_version.model_dump(), expanded.output_version.model_dump(),
                    compressed.output_version.model_dump()]
        before = json.dumps(versions, ensure_ascii=False, sort_keys=True)
        with tempfile.TemporaryDirectory() as tmp:
            logger = JsonlRunLogger(Path(tmp) / "runs.jsonl")
            request = TaskRequest(session_id="demo", user_goal="匿名职业规划")
            record = logger.log_run(request, response, feedback="过于详细", model_name="fake",
                                    adaptation=compressed, output_versions=list(reversed(versions)))
            self.assertEqual(record["output_versions"], versions)
            self.assertEqual(record["feedback_adaptation"]["source_version"], 2)
            self.assertEqual(logger.log_run(request, response, feedback="过于详细", model_name="fake",
                                            adaptation=compressed, output_versions=versions), record)
            self.assertEqual(len(logger.path.read_text().splitlines()), 1)
        self.assertEqual(json.dumps(versions, ensure_ascii=False, sort_keys=True), before)

    def test_incomplete_or_foreign_history_is_rejected_before_logging(self):
        response, expanded, compressed = two_rounds()
        versions = [response.output_version.model_dump(), expanded.output_version.model_dump(),
                    compressed.output_version.model_dump()]
        foreign = json.loads(json.dumps(versions))
        foreign[1]["plan_id"] = "foreign"
        for chain in (versions[1:], [versions[0], versions[2]], foreign):
            with self.subTest(chain=chain), tempfile.TemporaryDirectory() as tmp:
                logger = JsonlRunLogger(Path(tmp) / "runs.jsonl")
                with self.assertRaisesRegex(ValueError, "version history"):
                    logger.log_feedback_replay(compressed, chain)
                self.assertFalse(logger.path.exists())


class TestMachineIdentifierPrivacy(unittest.TestCase):
    @staticmethod
    def digest(length):
        prefix = "ab13812345678cd"
        return prefix + "a" * (length - len(prefix))

    def test_strict_machine_fields_keep_hashes_and_graph_references_intact(self):
        h20, h32, h64 = [self.digest(length) for length in (20, 32, 64)]
        payload = {
            "plan_id": h32, "request_id": h64, "facts_sha256": h64,
            "source_text_sha256": h64, "evidence_id": h20,
            "evidence_ids": [h20], "resolved_evidence_ids": [h20],
            "question_id": h20, "asked_keys": [h20],
            "session_id": "session-" + h64,
            "graph": {"id": "n_" + h20, "from": "n_" + h20,
                      "to": "n_" + h20, "provenance_ids": ["p_" + h20]},
        }
        self.assertNotEqual(redact_text(h64), h64)
        self.assertEqual(redact_data(payload), payload)
        self.assertEqual(redact_data({"request_id": h32})["request_id"], h32)
        self.assertEqual(session_key(payload["session_id"]), payload["session_id"])

    def test_known_key_names_do_not_exempt_raw_personal_information(self):
        payload = {
            "plan_id": "13812345678", "request_id": "demo@example.com",
            "facts_sha256": "联系13812345678", "source_text_sha256": "demo@example.com",
            "evidence_id": "13812345678", "session_id": "demo@example.com",
            "id": "13812345678", "from": "demo@example.com", "to": "13812345678",
            "evidence_ids": [self.digest(20), "13812345678"],
            "provenance_ids": ["p_" + self.digest(20), "demo@example.com"],
            # Even a valid digest is ordinary free text outside machine fields.
            "description": self.digest(64),
        }
        result = redact_data(payload)
        for key in ("plan_id", "request_id", "facts_sha256", "source_text_sha256",
                    "evidence_id", "session_id", "id", "from", "to", "description"):
            self.assertNotEqual(result[key], payload[key], key)
        self.assertEqual(result["evidence_ids"], [self.digest(20), "[REDACTED_PHONE]"])
        self.assertEqual(result["provenance_ids"], ["p_" + self.digest(20), "[REDACTED_EMAIL]"])

    def test_machine_ids_remain_stable_through_feedback_restart_and_replay(self):
        for backend in ("sqlite", "json"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sid = "session-" + self.digest(64)

                def make():
                    orch = CareerOrchestrator(
                        db_path=str(root / "sessions.db"), brain_client=AdaptableBrain(),
                        feedback_model=FakeFeedbackModel(), run_logger=JsonlRunLogger(root / "runs.jsonl"),
                    )
                    orch.memory.backend = backend
                    return orch

                orch = make()
                with patch("project.orchestrator.uuid.uuid4", return_value=SimpleNamespace(hex=self.digest(32))):
                    response = orch.run(TaskRequest(session_id=sid, user_goal="匿名职业规划"))
                self.assertEqual(response.plan_id, self.digest(32))
                self.assertEqual(orch.memory.get_latest_output_version(sid)["plan_id"], response.plan_id)
                token = self.digest(32)
                expanded = orch.adapt_feedback(sid, "过短", request_id=token)
                self.assertEqual(expanded.status, "adapted")
                restarted = make()
                replay = restarted.adapt_feedback(sid, "过短", request_id=token)
                self.assertEqual(replay, expanded)
                self.assertFalse(restarted.feedback_model.calls)
                compressed = restarted.adapt_feedback(sid, "过于详细", request_id=self.digest(64))
                self.assertEqual(compressed.status, "adapted")
                self.assertEqual(compressed.output_version.version, 3)
                self.assertEqual(restarted.memory.get_feedback_result(self.digest(64))["request_id"], self.digest(64))
                records = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertEqual(records[-1]["output_versions"][-1]["plan_id"], response.plan_id)


if __name__ == "__main__":
    unittest.main()
