from __future__ import annotations

import unittest

from project.core.intake import FOLLOW_UP_QUESTIONS, apply_answers_to_profile, collect_follow_up_answers
from project.core.schemas import UserProfile


class TestFollowUpIntake(unittest.TestCase):
    def test_question_sequence_has_eight_stable_profile_dimensions(self) -> None:
        self.assertEqual(
            [question.key for question in FOLLOW_UP_QUESTIONS],
            [
                "education",
                "major",
                "skills",
                "interests",
                "target_role",
                "time_budget",
                "preference",
                "constraints",
            ],
        )
        self.assertEqual(len(FOLLOW_UP_QUESTIONS), 8)

    def test_collector_asks_every_question_in_order(self) -> None:
        answers = iter(["本科大三", "计算机", "Python、SQL", "数据分析", "数据分析师", "每周10小时", "上海互联网", "缺少项目经验"])
        prompts: list[str] = []

        result = collect_follow_up_answers(
            input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        )

        self.assertEqual(len(prompts), 8)
        self.assertEqual(result["education"], "本科大三")
        self.assertEqual(result["constraints"], "缺少项目经验")

    def test_answers_are_merged_into_structured_profile(self) -> None:
        profile = apply_answers_to_profile(
            UserProfile(strengths=["已有项目经历"]),
            {
                "education": "本科大三",
                "major": "计算机科学",
                "skills": "Python、SQL、Excel",
                "interests": "数据分析，人工智能",
                "target_role": "数据分析师",
                "time_budget": "每周 12 小时",
                "preference": "上海或互联网行业",
                "constraints": "英语薄弱；预算有限",
            },
        )

        self.assertEqual(profile.education_stage, "本科大三")
        self.assertEqual(profile.major, "计算机科学")
        self.assertEqual(profile.skills, ["Python", "SQL", "Excel"])
        self.assertEqual(profile.interests, ["数据分析", "人工智能"])
        self.assertEqual(profile.target_role, "数据分析师")
        self.assertEqual(profile.constraints.time_budget_hours_per_week, 12)
        self.assertEqual(profile.preference, "上海或互联网行业")
        self.assertEqual(profile.main_constraints, ["英语薄弱", "预算有限"])
        self.assertEqual(profile.strengths, ["已有项目经历"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
