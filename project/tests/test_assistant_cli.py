from __future__ import annotations

import json
import unittest

from project import assistant_cli


class FakeResponse:
    def model_dump(self):
        return {"target_roles": ["数据分析师"], "served_by": "cloud_brain"}


class FakeOrchestrator:
    def __init__(self) -> None:
        self.request = None
        self.feedback = None

    def run(self, request):
        self.request = request
        return FakeResponse()

    def submit_feedback(self, session_id, feedback):
        self.feedback = (session_id, feedback)


class TestAssistantCli(unittest.TestCase):
    def test_default_cli_collects_all_follow_up_answers_before_planning(self) -> None:
        fake = FakeOrchestrator()
        inputs = iter(["本科大三", "统计学", "Python、SQL", "数据分析", "数据分析师", "10", "上海", "项目不足", "2"])
        output: list[str] = []

        exit_code = assistant_cli.main(
            ["--goal", "获得数据分析实习", "--text", "会 Python"],
            input_fn=lambda _prompt: next(inputs),
            output_fn=output.append,
            orchestrator_factory=lambda: fake,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(fake.request.follow_up_answers), 8)
        self.assertEqual(fake.request.follow_up_answers["major"], "统计学")
        self.assertEqual(fake.feedback, ("default-session", "合适"))
        self.assertTrue(any("数据分析师" in line for line in output))

    def test_answers_json_makes_cli_repeatable_without_prompts(self) -> None:
        fake = FakeOrchestrator()
        answers = {
            "education": "本科",
            "major": "软件工程",
            "skills": "Python",
            "interests": "后端开发",
            "target_role": "后端开发工程师",
            "time_budget": "8",
            "preference": "杭州互联网",
            "constraints": "实习经历少",
        }

        exit_code = assistant_cli.main(
            ["--goal", "求职", "--answers-json", json.dumps(answers, ensure_ascii=False)],
            input_fn=lambda _prompt: "3",
            output_fn=lambda _line: None,
            orchestrator_factory=lambda: fake,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake.request.follow_up_answers, answers)
        self.assertEqual(fake.feedback, ("default-session", "过于详细"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
