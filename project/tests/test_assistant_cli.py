from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project import assistant_cli
from project.core.run_logging import JsonlRunLogger
from project.orchestrator import CareerOrchestrator
from project.tests.test_completion_flow import FakeDeepSeekClient
from project.tests.test_guidance import FakeGuidanceModel
from project.tests.test_output_adaptation import FakeFeedbackModel


class TestAssistantCli(unittest.TestCase):
    def run_cli(self, args, inputs, model=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orchestrator = CareerOrchestrator(
                db_path=str(root / "sessions.db"), brain_client=FakeDeepSeekClient(),
                guidance_model=model or FakeGuidanceModel(), feedback_model=FakeFeedbackModel(), run_logger=JsonlRunLogger(root / "runs.jsonl"),
            )
            orchestrator.settings = orchestrator.settings.model_copy(update={"guidance_max_rounds": 2})
            iterator = iter(inputs)
            prompts, output = [], []
            with patch.object(orchestrator, 'run', wraps=orchestrator.run) as run:
                code = assistant_cli.main(
                    args, input_fn=lambda prompt: prompts.append(prompt) or next(iterator),
                    output_fn=output.append, orchestrator_factory=lambda: orchestrator,
                )
            history = orchestrator.memory.get_session_history("default-session")
            return code, prompts, output, history, run.call_count

    def test_default_cli_collects_only_targeted_questions_before_planning(self):
        code, prompts, output, history, calls = self.run_cli(
            ["--goal", "求职", "--text", "会 Python"], ["数据分析师", "SQL", "2"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(calls, 1)
        self.assertEqual(len(prompts), 3)  # Two guidance questions and feedback.
        self.assertEqual(history[-1]["request"]["follow_up_answers"], {"target_role": "数据分析师", "skills": "SQL"})
        self.assertTrue(any("数据分析师" in line for line in output))
        self.assertNotIn("question_id", "".join(output))
        self.assertIn("question_id", history[-1]["response"]["profile"]["guidance"]["answers"][0])

    def test_complete_answers_json_needs_no_guidance_questions(self):
        answers = {
            "education": "本科", "major": "软件工程", "skills": "Python", "interests": "后端开发",
            "target_role": "后端开发工程师", "time_budget": "8", "preference": "杭州互联网", "constraints": "实习经历少",
        }
        code, prompts, _, history, _ = self.run_cli(
            ["--goal", "求职", "--answers-json", json.dumps(answers, ensure_ascii=False)], ["3"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(history[-1]["request"]["follow_up_answers"], answers)

    def test_partial_json_does_not_skip_missing_questions(self):
        code, prompts, _, history, _ = self.run_cli(
            ["--goal", "求职", "--answers-json", '{"target_role":"数据分析师"}'], ["SQL", "2.5", "2"],
        )
        self.assertEqual(code, 0)
        self.assertIn("技能", prompts[0])
        self.assertEqual(history[-1]["response"]["profile"]["constraints"]["time_budget_hours_per_week"], 2.5)

    def test_skip_preserves_provided_answers_without_local_calls(self):
        model = FakeGuidanceModel()
        code, prompts, _, history, _ = self.run_cli(
            ["--goal", "求职", "--no-follow-up", "--answers-json", '{"skills":"无"}'], ["2"], model,
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(prompts), 1)
        self.assertFalse(model.contexts)
        self.assertEqual(history[-1]["response"]["profile"]["guidance"]["fields"]["skills"]["status"], "explicit_none")

    def test_model_failure_requires_explicit_continue_before_planning(self):
        for choice, expected in [("退出", 1), ("继续", 0)]:
            with self.subTest(choice=choice):
                code, _, output, history, calls = self.run_cli(
                    ["--goal", "求职"], [choice, "2"], FakeGuidanceModel([RuntimeError("secret-path")]),
                )
                self.assertEqual(code, expected)
                self.assertEqual(calls, int(choice == "继续"))
                self.assertNotIn("secret-path", "".join(output))
                if history:
                    self.assertEqual(history[-1]["response"]["profile"]["guidance"]["stop_reason"], "model_error")

    def test_bad_answer_fields_and_invalid_hours_fail_before_orchestrator_creation(self):
        for args in [["--answers-json", '{"unknown":"x"}'], ["--answers-json", '{"skills":null}'],
                     ["--time-budget", "-1"], ["--time-budget", "nan"]]:
            with self.subTest(args=args):
                with patch.object(assistant_cli, 'CareerOrchestrator') as factory:
                    code = assistant_cli.main(["--goal", "求职", *args],
                                              orchestrator_factory=factory, output_fn=lambda _: None)
                    self.assertEqual(code, 2)
                    factory.assert_not_called()
