"""Independent regressions for persisted provenance and privacy projections."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.knowledge_versions import KnowledgeRegistry
from project.core.knowledge_graph import build_career_graph
from project.core.output_adaptation import facts_hash
from project.core.schemas import CareerPlanResponse, PerceptionResult
from project.core.web_workflow import WebPlanningEngine, WorkflowError
from project.orchestrator import CareerOrchestrator
from project.tests.test_completion_flow import FakeDeepSeekClient
from project.tests.test_guidance import FakeGuidanceModel
from project.utils.fusion import MultiModalFusion


class TestWebSnapshotReview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = KnowledgeRegistry(self.root / "knowledge")

    def make_engine(self, model=None, *, max_rounds=0):
        orchestrator = CareerOrchestrator(
            persist=False, guidance_model=model or FakeGuidanceModel(),
            brain_client=FakeDeepSeekClient(),
        )
        return WebPlanningEngine(orchestrator, self.registry, max_rounds=max_rounds)

    def new_state(self):
        return {"id": "independent-web-review", "user_goal": "寻找数据分析实习", "text_input": ""}

    def test_machine_knowledge_version_survives_the_planning_privacy_projection(self):
        snapshot = self.registry.snapshot("数据分析")
        # This is an opaque 20-digit hexadecimal release suffix, not a phone.
        snapshot["version_id"] = "kb-a13812345678bccccccc"
        self.assertEqual(len(snapshot["version_id"].split("-", 1)[1]), 20)
        engine = self.make_engine()
        with patch.object(self.registry, "snapshot", return_value=copy.deepcopy(snapshot)):
            state = engine.process("perceive", self.new_state(), {}, [])
            state = engine.process("plans", state, {}, [])
        self.assertEqual(state["knowledge_snapshot"]["version_id"], snapshot["version_id"])
        response = CareerPlanResponse.model_validate(state["plan"])
        self.assertEqual(state["versions"][-1]["facts_sha256"], facts_hash(response))

    def test_upload_quotes_pii_and_paths_do_not_break_saved_feedback_hashes(self):
        file = self.root / "resume.txt"
        file.write_text("我掌握SQL，联系 demo@example.com。", encoding="utf-8")
        engine = self.make_engine()
        state = engine.process("perceive", self.new_state(), {}, [{
            "id": "a13812345678bccccccc", "path": str(file), "name": "resume.txt", "subject": "student",
        }])
        state = engine.process("plans", state, {}, [])
        response = CareerPlanResponse.model_validate(state["plan"])
        self.assertEqual(state["versions"][-1]["facts_sha256"], facts_hash(response))
        restored = engine.process("feedback", state, {
            "plan_id": state["plan"]["plan_id"], "source_version": 1, "feedback": "合适",
        }, [])
        self.assertEqual(restored["feedback"][-1]["status"], "unchanged")
        self.assertEqual(restored["feedback"][-1]["reason"], "suitable")
        self.assertEqual(restored["feedback"][-1]["output_version"]["facts_sha256"], facts_hash(response))

    def test_next_guidance_failure_preserves_the_submitted_answer_in_partial_state(self):
        engine = self.make_engine(FakeGuidanceModel(), max_rounds=4)
        state = engine.process("perceive", self.new_state(), {}, [])
        engine.orchestrator.guidance_model = FakeGuidanceModel([RuntimeError("anonymous failure")])
        with patch.object(engine.orchestrator, "perceive", side_effect=AssertionError("perception must not repeat")):
            with self.assertRaises(WorkflowError) as caught:
                engine.process("answers", state, {
                    "question_id": state["question"]["id"], "answer": "数据分析师",
                }, [])
        saved = caught.exception.partial_state
        self.assertTrue(saved["extracted"])
        self.assertEqual(saved["rounds"], 1)
        self.assertEqual(saved["profile"]["target_role"], "数据分析师")
        self.assertEqual(saved["profile"]["guidance"]["answers"][-1]["raw_answer"], "数据分析师")
        self.assertEqual(saved["status"], "ready")

    def test_non_student_facts_are_not_reintroduced_as_unscoped_merged_facts(self):
        fused = MultiModalFusion.fuse([PerceptionResult(
            modality="document", subject="job", verification="unverified", source_id="job-1",
            summary="岗位说明", facts=["掌握 Kubernetes"], confidence=0.9,
        )])
        for line in fused.splitlines():
            if line.startswith("[merged_facts]"):
                self.assertNotIn("掌握 Kubernetes", line)

    def test_unknown_or_job_text_is_not_given_to_student_guidance(self):
        for subject in ("unknown", "job", "reference"):
            with self.subTest(subject=subject):
                model = FakeGuidanceModel()
                engine = self.make_engine(model, max_rounds=1)
                state = {**self.new_state(), "text_input": "岗位要求掌握 Kubernetes", "text_subject": subject}
                result = engine.process("perceive", state, {}, [])
                self.assertTrue(all("Kubernetes" not in item["excerpt"] for item in model.contexts[0]["evidence"]))
                self.assertEqual(result["profile"]["skills"], [])
                material = next(claim for claim in result["source_claims"] if claim["field"] == "material")
                self.assertEqual(material["subject"], subject)
                self.assertEqual(material["verification"], "unverified")

    def test_unverified_student_document_keeps_its_status_in_graph_provenance(self):
        file = self.root / "student.txt"
        file.write_text("我掌握SQL。", encoding="utf-8")

        def observe_document(context):
            item = next(e for e in context["evidence"] if e["origin"].endswith(":parsed_text"))
            return {"observations": [{"field": "skills", "value": "SQL", "evidence_ids": [item["evidence_id"]]}]}

        model = FakeGuidanceModel([observe_document])
        state = self.make_engine(model, max_rounds=1).process("perceive", self.new_state(), {}, [{
            "id": "student-source", "path": str(file), "name": "student.txt", "subject": "student",
        }])
        response = CareerPlanResponse(session_id="review", plan_id="a" * 32, intent="planning", profile=state["profile"])
        graph = build_career_graph(response)
        source = next(item for item in graph["provenance"] if item.get("source_id") == "student-source")
        self.assertEqual(source["subject"], "student")
        self.assertEqual(source["verification"], "unverified")
        self.assertEqual(source["status"], "unverified_input")


if __name__ == "__main__":
    unittest.main()
