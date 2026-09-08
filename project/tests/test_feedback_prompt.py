from __future__ import annotations

import unittest

from project.core.feedback_prompt import (
    FEEDBACK_PROMPT_VERSION,
    TIER_INSTRUCTIONS,
    build_feedback_prompt,
)


class TestFeedbackPrompt(unittest.TestCase):
    def _build(self, feedback: str = "过短") -> str:
        return build_feedback_prompt(
            original_plan_json='{"target_roles": ["数据分析师"], "confidence": 0.8}',
            feedback=feedback,
            user_goal="获得数据分析实习",
            constraints_json='{"city": "上海"}',
            profile_json='{"strengths": ["Python"]}',
            knowledge_hints=["数据分析师 | 核心技能: SQL | 薪资参考: 10k"],
        )

    def test_version_is_feedback_v1(self) -> None:
        self.assertEqual(FEEDBACK_PROMPT_VERSION, "feedback-v1")

    def test_prompt_contains_strict_json_schema_and_requirements(self) -> None:
        prompt = self._build()
        self.assertIn("请只输出严格 JSON", prompt)
        self.assertIn('"user_facing_advice"', prompt)
        self.assertIn('"roadmap_30_90_180"', prompt)
        self.assertIn("不得虚构知识库未提供的职业事实", prompt)

    def test_short_tier_instructs_expansion(self) -> None:
        prompt = self._build("过短")
        self.assertIn("反馈档位: 过短", prompt)
        self.assertIn("不少于 4 条", prompt)
        self.assertIn("1.5 倍", prompt)

    def test_detailed_tier_instructs_condensing(self) -> None:
        prompt = self._build("过于详细")
        self.assertIn("反馈档位: 过于详细", prompt)
        self.assertIn("不超过 200 字", prompt)
        self.assertIn("不超过 3 条", prompt)

    def test_tier_instructions_differ(self) -> None:
        self.assertNotEqual(TIER_INSTRUCTIONS["过短"], TIER_INSTRUCTIONS["过于详细"])

    def test_prompt_embeds_original_plan_and_context(self) -> None:
        prompt = self._build()
        self.assertIn('{"target_roles": ["数据分析师"], "confidence": 0.8}', prompt)
        self.assertIn("获得数据分析实习", prompt)
        self.assertIn("数据分析师 | 核心技能: SQL", prompt)

    def test_unsupported_feedback_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._build("合适")
        with self.assertRaises(ValueError):
            self._build("乱写")


if __name__ == "__main__":
    unittest.main()
