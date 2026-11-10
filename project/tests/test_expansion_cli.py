from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from project import assistant_cli
from project.core.run_logging import JsonlRunLogger
from project.core.report_export import export_report
from project.core.privacy import session_key
from project.core.schemas import PerceptionResult, TaskRequest
from project.orchestrator import CareerOrchestrator
from project.tests.test_output_adaptation import AdaptableBrain, FakeFeedbackModel


class TestExpansionCli(unittest.TestCase):
    def make_orchestrator(self, root, *, long=False):
        return CareerOrchestrator(
            db_path=str(root / "sessions.db"), brain_client=AdaptableBrain(long),
            feedback_model=FakeFeedbackModel(), run_logger=JsonlRunLogger(root / "runs.jsonl"),
        )

    def test_cli_three_versions_report_and_restart_without_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.make_orchestrator(root, long=True)
            inputs, output = iter(["3", "1", "结束"]), []
            code = assistant_cli.main(
                ["--goal", "数据分析师", "--no-follow-up", "--feedback-rounds", "3",
                 "--export-dir", str(root / "report")],
                input_fn=lambda _: next(inputs), output_fn=output.append,
                orchestrator_factory=lambda: orch,
            )
            self.assertEqual(code, 0)
            self.assertEqual(orch.cloud_brain.calls, 1)
            response = orch.get_saved_plan("default-session")
            versions = orch.memory.get_output_versions("default-session", response.plan_id)
            self.assertEqual([v["version"] for v in versions], [1, 2, 3])
            records = [json.loads(line) for line in (root / "runs.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["output_versions"], versions)
            for name in ["report.md", "graph.mmd", "graph.json"]:
                self.assertTrue((root / "report" / name).is_file())
            self.assertIn(versions[-1]["display_text"], (root / "report" / "report.md").read_text())

            restarted = self.make_orchestrator(root)
            restarted.cloud_brain.plan = MagicMock(side_effect=AssertionError("no new planning"))
            code = assistant_cli.main(
                ["--resume-feedback", "--export-dir", str(root / "recovered")],
                input_fn=lambda _: "2", output_fn=output.append,
                orchestrator_factory=lambda: restarted,
            )
            self.assertEqual(code, 0)
            restarted.cloud_brain.plan.assert_not_called()
            self.assertFalse(restarted.feedback_model.calls)
            self.assertIn("第 3 版", "".join(output))

    def test_restart_with_personal_input_and_document_evidence(self):
        for backend in ["sqlite", "json"]:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                document = root / "profile.txt"
                document.write_text("本科统计学，会Python，联系 demo@example.com", encoding="utf-8")
                orch = self.make_orchestrator(root)
                orch.memory.backend = backend
                response = orch.run(TaskRequest(
                    session_id="s", user_goal="数据分析师", document_paths=[str(document)],
                    follow_up_answers={"constraints": "联系 demo@example.com 或 13812345678"},
                ))
                first = orch.adapt_feedback("s", "过短", request_id="first")
                self.assertEqual(first.status, "adapted")
                restarted = self.make_orchestrator(root)
                restarted.memory.backend = backend
                restored = restarted.get_saved_plan("s", response.plan_id)
                self.assertEqual(restored.plan_id, response.plan_id)
                second = restarted.adapt_feedback("s", "合适", request_id="second")
                self.assertEqual(second.source_version, 2)
                self.assertEqual(second.status, "unchanged")
                self.assertEqual(second.output_version, first.output_version)
                self.assertNotIn("demo@example.com", (root / "runs.jsonl").read_text())
                with self.assertRaises(ValueError):
                    restarted.adapt_feedback("other-session", "过短", plan_id=response.plan_id)

    def test_video_and_scan_are_explicit_and_regular_documents_stay_model_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            orch.document_agent = MagicMock()
            orch.document_agent.perceive.return_value = PerceptionResult(modality="document", summary="doc")
            video = MagicMock()
            video.perceive.return_value = PerceptionResult(modality="video", summary="video")
            with patch.object(orch, "_get_image_agent") as image_factory, \
                    patch.object(orch, "_get_video_agent", return_value=video):
                orch._collect_perception(TaskRequest(session_id="s", user_goal="求职", document_paths=["normal.pdf"]))
                image_factory.assert_not_called()
                orch._collect_perception(TaskRequest(
                    session_id="s", user_goal="求职", document_paths=["scanned.pdf"],
                    video_paths=["talk.mp4"], metadata={"scan_pdf": "true"},
                ))
                orch.document_agent.perceive.assert_called_with("scanned.pdf", image_agent=image_factory.return_value)
                video.perceive.assert_called_once_with("talk.mp4")
            parsed = assistant_cli.build_parser().parse_args(["--goal", "求职", "--video", "talk.mp4", "--scan-pdf"])
            self.assertEqual(parsed.video, ["talk.mp4"])
            self.assertTrue(parsed.scan_pdf)

    def test_invalid_options_and_existing_export_stop_before_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in [[], ["--goal", "求职", "--feedback-rounds", "0"],
                         ["--goal", "求职", "--feedback-rounds", "17"],
                         ["--goal", "求职", "--plan-id", "foreign"],
                         ["--goal", "求职", "--export-dir", tmp]]:
                with self.subTest(args=args):
                    factory = MagicMock()
                    self.assertEqual(assistant_cli.main(args, orchestrator_factory=factory, output_fn=lambda _: None), 2)
                    factory.assert_not_called()

    def test_complete_knowledge_contents_reach_actual_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            captured = []
            original_plan = orch.cloud_brain.plan
            def plan(prompt, model=None):
                captured.append(prompt)
                return original_plan(prompt, model)
            orch.cloud_brain.plan = plan
            response = orch.run(TaskRequest(session_id="s", user_goal="数据分析师", skip_follow_up=True))
            self.assertTrue(response.knowledge_hit_ids)
            for field in ["SQLBolt", "career-", "核验状态: not_verified"]:
                self.assertIn(field, captured[0])

    def test_personal_session_identifier_maps_consistently_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_id = "demo@example.com"
            orch = self.make_orchestrator(root)
            response = orch.run(TaskRequest(session_id=original_id, user_goal="数据分析师"))
            self.assertEqual(response.session_id, session_key(original_id))
            first = orch.adapt_feedback(original_id, "过短", request_id="expand")
            self.assertEqual(first.status, "adapted")
            restarted = self.make_orchestrator(root)
            self.assertEqual(restarted.get_saved_plan(original_id).plan_id, response.plan_id)
            second = restarted.adapt_feedback(original_id, "合适", request_id="keep")
            self.assertEqual(second.source_version, 2)
            self.assertNotIn(original_id, (root / "runs.jsonl").read_text())

    def test_export_invalid_encoding_never_reserves_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            response = self.make_orchestrator(root).run(TaskRequest(session_id="s", user_goal="求职"))
            with self.assertRaises(UnicodeError):
                export_report(response, [], root / "invalid", display_text="bad\ud800")
            self.assertFalse((root / "invalid").exists())

    def test_scanned_interpretation_is_not_promoted_to_user_evidence_or_strengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            derived = PerceptionResult(
                modality="document", summary="visual", raw_output="熟悉Python，专业统计学",
                facts=["熟悉Python，专业统计学"], requires_confirmation=True,
            )
            req = TaskRequest(session_id="s", user_goal="求职", skip_follow_up=True)
            with patch.object(orch, "_collect_perception", return_value=[derived]):
                prepared = orch.prepare(req)
            self.assertEqual(prepared.profile.strengths, [])
            self.assertEqual(prepared.profile.guidance.fields["major"].status, "missing")
            self.assertFalse(any("统计学" in item.excerpt for item in prepared.profile.guidance.evidence))
            prompt = orch._build_planning_prompt(req, "planning", prepared.profile, [derived], [])
            self.assertIn("待核实", prompt)

    def test_visual_agent_is_unloaded_after_material_collection_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            orch.image_agent = MagicMock()
            orch.document_agent = MagicMock()
            orch.document_agent.perceive.side_effect = RuntimeError("failed material")
            with self.assertRaises(RuntimeError):
                orch._collect_perception(TaskRequest(session_id="s", user_goal="求职", document_paths=["bad.pdf"]))
            orch.image_agent.unload.assert_called_once()

    def test_uuid_with_phone_shaped_digits_survives_persistence_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.make_orchestrator(root)
            machine_id = "ab13812345678cd12ef34ab56cd78ef9"
            self.assertEqual(len(machine_id), 32)
            with patch("project.orchestrator.uuid.uuid4") as uuid_factory:
                uuid_factory.return_value.hex = machine_id
                response = orch.run(TaskRequest(session_id="s", user_goal="求职"))
            self.assertEqual(response.plan_id, machine_id)
            self.assertEqual(orch.memory.get_latest_output_version("s")["plan_id"], machine_id)
            self.assertEqual(orch.adapt_feedback("s", "过短", request_id="token").status, "adapted")

    def test_failed_material_and_late_truncation_warning_reach_planning_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            req = TaskRequest(session_id="s", user_goal="求职")
            failed = PerceptionResult(modality="document", summary="failed", confidence=0,
                                      missing_info=["Document parsing failed"])
            limited = PerceptionResult(modality="document", summary="partial", confidence=0.6,
                                       missing_info=["warning"] * 5 + ["Document text truncated"])
            profile = orch._build_profile(req, [failed, limited])
            prompt = orch._build_planning_prompt(req, "planning", profile, [failed, limited], [])
            self.assertIn("Document parsing failed", prompt)
            self.assertIn("Document text truncated", prompt)


if __name__ == "__main__":
    unittest.main()
