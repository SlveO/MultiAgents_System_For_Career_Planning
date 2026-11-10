from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.brain_client import BrainTimeoutError
from project.core.guidance import advance_guidance, submit_guidance_answer
from project.core.intake import initialize_fields
from project.core.knowledge_versions import KnowledgeRegistry
from project.core.output_adaptation import facts_hash
from project.core.report_export import export_report
from project.core.schemas import CareerPlanResponse, GuidanceDecision, UserProfile, TaskRequest
from project.core.web_workflow import WebPlanningEngine, WorkflowError
from project.orchestrator import CareerOrchestrator
from project.tests.test_completion_flow import FakeDeepSeekClient
from project.tests.test_guidance import FakeGuidanceModel


class CatalogFeedback:
    def __init__(self):
        self.calls = 0

    def analyze(self, context):
        self.calls += 1
        selected = set(context["required_ids"])
        groups = {}
        for item in context["catalog"]:
            if context["feedback"] == "过短" or item["item_id"] in selected:
                groups.setdefault(item["section"], []).append(item["item_id"])
        return {"sections": [{"section": key, "item_ids": ids, "style": "bullets"} for key, ids in groups.items()]}

    def unload(self):
        pass


class TestWebWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = KnowledgeRegistry(self.root / "knowledge")

    def tearDown(self):
        self.temp.cleanup()

    def engine(self, model=None, brain=None, max_rounds=4):
        orch = CareerOrchestrator(
            persist=False, guidance_model=model or FakeGuidanceModel(),
            brain_client=brain or FakeDeepSeekClient(), feedback_model=CatalogFeedback(),
        )
        return WebPlanningEngine(orch, self.registry, max_rounds=max_rounds)

    def state(self):
        return {"id": "web-anonymous-test", "user_goal": "寻找数据分析实习", "text_input": "", "status": "collecting"}

    def test_jobs_do_not_become_student_abilities_and_sources_stay_unverified(self):
        file = self.root / "requirements.txt"
        file.write_text("岗位要求熟悉 Kubernetes 和 Go，有三年项目经验。")
        model = FakeGuidanceModel()
        engine = self.engine(model)
        state = engine.process("perceive", self.state(), {}, [{
            "id": "upload-a", "path": str(file), "name": "requirements.txt", "subject": "job",
        }])
        self.assertEqual(state["profile"]["skills"], [])
        self.assertEqual(state["profile"]["strengths"], [])
        self.assertTrue(all("Kubernetes" not in e["excerpt"] for e in model.contexts[0]["evidence"]))
        job_claim = next(claim for claim in state["source_claims"] if claim["subject"] == "job")
        self.assertEqual(job_claim["verification"], "unverified")
        self.assertNotIn(str(self.root), json.dumps(state))
        self.assertEqual(state["request"]["document_paths"], [])

    def test_restart_answers_resume_saved_question_without_perceiving(self):
        state = self.engine().process("perceive", self.state(), {}, [])
        state = json.loads(json.dumps(state))
        restart = self.engine()
        with patch.object(restart.orchestrator, "perceive", side_effect=AssertionError("must not re-perceive")):
            saved = copy.deepcopy(state)
            next_state = restart.process("answers", state, {"question_id": state["question"]["id"], "answer": "数据分析师"}, [])
        self.assertEqual(state, saved)
        self.assertEqual(next_state["profile"]["target_role"], "数据分析师")
        self.assertEqual(next_state["rounds"], 1)
        self.assertNotEqual(next_state["question"]["id"], state["question"]["id"])
        with self.assertRaises(WorkflowError):
            restart.process("answers", next_state, {"question_id": state["question"]["id"], "answer": "重复"}, [])

    def test_guidance_failure_preserves_extraction_and_answer_without_hidden_retry(self):
        model = FakeGuidanceModel([RuntimeError("private path")])
        engine = self.engine(model)
        with self.assertRaises(WorkflowError) as caught:
            engine.process("perceive", self.state(), {}, [])
        state = caught.exception.partial_state
        self.assertTrue(state["extracted"])
        self.assertEqual(state["status"], "ready")
        self.assertEqual(len(model.contexts), 1)
        with patch.object(engine.orchestrator, "perceive", side_effect=AssertionError("must not repeat")):
            planned = engine.process("plans", state, {}, [])
        self.assertEqual(planned["status"], "planned")

    def test_four_answers_end_guidance_and_unknown_is_not_reasked(self):
        engine = self.engine()
        state = engine.process("perceive", self.state(), {}, [])
        fields = []
        for answer in ["数据分析师", "不知道", "8", "本科大三"]:
            fields.append(state["question"]["field"])
            state = engine.process("answers", state, {"question_id": state["question"]["id"], "answer": answer}, [])
        self.assertEqual(len(set(fields)), 4)
        self.assertEqual(state["profile"]["guidance"]["fields"]["skills"]["status"], "unknown")
        self.assertIsNone(state["question"])
        self.assertEqual(state["profile"]["guidance"]["stop_reason"], "round_limit")
        self.assertEqual(len(engine.orchestrator.guidance_model.contexts), 4)

    def test_snapshot_report_and_feedback_survive_new_release_and_restart(self):
        engine = self.engine(max_rounds=0)
        state = engine.process("perceive", self.state(), {}, [])
        state = engine.process("plans", state, {}, [])
        old_snapshot = copy.deepcopy(state["knowledge_snapshot"])
        original_report = state["plan"]["report_markdown"]
        original_response = CareerPlanResponse.model_validate(state["plan"])
        entries = self.registry.current()["entries"]
        entries[0]["skills"].append("SHOULD_NOT_APPEAR_IN_OLD_REPORT")
        candidate = self.registry.create_candidate(entries, author="test")
        self.registry.review(candidate["id"], reviewer="reviewer", approve=True, note="test")
        self.registry.publish(candidate["id"], actor="test")
        report_dir = export_report(original_response, entries, self.root / "export")
        self.assertNotIn("SHOULD_NOT_APPEAR", (report_dir / "report.md").read_text())
        restart = self.engine()
        for choice in ["过短", "过于详细", "合适"]:
            latest = state["versions"][-1]["version"]
            state = restart.process("feedback", json.loads(json.dumps(state)), {
                "plan_id": state["plan"]["plan_id"], "source_version": latest, "feedback": choice,
            }, [])
        self.assertEqual(state["knowledge_snapshot"], old_snapshot)
        self.assertEqual(len(state["versions"]), 3)
        self.assertEqual(restart.orchestrator.feedback_model.calls, 1)
        self.assertTrue(all(version["facts_sha256"] == facts_hash(original_response) for version in state["versions"]))
        self.assertEqual(state["plan"]["user_facing_advice"], original_response.user_facing_advice)
        self.assertNotIn("SHOULD_NOT_APPEAR", state["plan"]["report_markdown"])
        self.assertTrue(original_report)

    def test_brief_view_survives_serialization_and_full_markdown_export(self):
        engine = self.engine(max_rounds=0)
        state = engine.process("perceive", self.state(), {}, [])
        state = engine.process("plans", state, {}, [])
        full_text = state["versions"][-1]["display_text"]
        state = engine.process("feedback", state, {
            "plan_id": state["plan"]["plan_id"], "source_version": 1,
            "feedback": "过于详细",
        }, [])
        restored = json.loads(json.dumps(state, ensure_ascii=False))
        self.assertEqual(restored["versions"][-1]["reading_view"]["mode"], "brief")
        self.assertEqual(restored["versions"][-1]["display_text"], full_text)
        self.assertIn(full_text, restored["plan"]["report_markdown"])
        self.assertEqual(engine.orchestrator.feedback_model.calls, 0)

    def test_brain_failure_is_one_request_and_no_template_success(self):
        class FailedBrain:
            calls = 0

            def plan(self, *args, **kwargs):
                self.calls += 1
                raise BrainTimeoutError("timeout")

        brain = FailedBrain()
        engine = self.engine(brain=brain, max_rounds=0)
        state = engine.process("perceive", self.state(), {}, [])
        with self.assertRaisesRegex(WorkflowError, "未自动重试"):
            engine.process("plans", state, {}, [])
        self.assertEqual(brain.calls, 1)
        self.assertIsNone(state["plan"])

    def test_web_orchestrator_never_opens_cli_storage_or_logger(self):
        with patch("project.orchestrator.SessionMemory") as memory, patch("project.orchestrator.JsonlRunLogger") as logger:
            self.engine(max_rounds=0).process("perceive", self.state(), {}, [])
        memory.assert_not_called()
        logger.assert_not_called()

    def test_legacy_export_does_not_enrich_history_from_current_knowledge(self):
        engine = self.engine(max_rounds=0)
        state = engine.process("perceive", self.state(), {}, [])
        state = engine.process("plans", state, {}, [])
        response = CareerPlanResponse.model_validate(state["plan"])
        response.knowledge_snapshot = None
        current_items = [{"item_id": item, "role": "CURRENT_RELEASE_MUST_NOT_LEAK"} for item in response.knowledge_hit_ids]
        target = export_report(response, current_items, self.root / "legacy-report")
        report = (target / "report.md").read_text()
        self.assertNotIn("CURRENT_RELEASE_MUST_NOT_LEAK", report)
        self.assertIn("没有知识快照", report)

    def test_cli_reads_reviewed_knowledge_directory_from_dotenv_settings(self):
        from project.core.settings import AppSettings

        dotenv = self.root / "config.env"
        dotenv.write_text(f"CAREER_KNOWLEDGE_DIR={self.registry.root}\n")
        orch = CareerOrchestrator(persist=False, brain_client=FakeDeepSeekClient())
        orch.settings = AppSettings(_env_file=dotenv)
        response = orch.run(TaskRequest(session_id="cli-config-test", user_goal="数据分析师"))
        self.assertEqual(response.knowledge_snapshot["version_id"], self.registry.current()["version_id"])

    def test_single_step_answer_is_bound_to_question_without_terminal_input(self):
        profile = UserProfile()
        initialize_fields(profile)
        req = TaskRequest(session_id="step", user_goal="规划")
        step = advance_guidance(req, profile, FakeGuidanceModel(), round_number=1, max_rounds=4)
        with self.assertRaisesRegex(ValueError, "stale"):
            submit_guidance_answer(req, step.profile, step.question, "old-id", "Python", round_number=1)
        profile = submit_guidance_answer(req, step.profile, step.question, step.question_id, "不知道", round_number=1)
        answer = profile.guidance.answers[-1]
        self.assertEqual(answer.status, "unknown")
        self.assertEqual(profile.guidance.evidence[-1].verification, "user_confirmed")


if __name__ == "__main__":
    unittest.main()
