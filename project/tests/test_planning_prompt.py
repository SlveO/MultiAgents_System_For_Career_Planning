"""Planning-prompt contract tests (member B, week 2).

Covers the manual's acceptance criteria for the planning prompt: direction,
gap analysis, and phased actions are required; career facts must cite the
injected knowledge hints; unsupported input is flagged instead of invented.
"""

import unittest

from project.core.planning_prompt import (
    PLANNING_PROMPT_VERSION,
    build_planning_prompt,
)

GOAL = "获得数据分析实习"
TEXT = "会 Python 和 SQL"
INTENT = "planning"
CONSTRAINTS = "{}"
PROFILE = '{"major": "计算机", "grade": "大三"}'
PERCEPTION = "无感知结果"


def _render(knowledge_hints=None):
    return build_planning_prompt(
        user_goal=GOAL,
        text_input=TEXT,
        intent=INTENT,
        constraints_json=CONSTRAINTS,
        profile_json=PROFILE,
        perception_text=PERCEPTION,
        knowledge_hints=knowledge_hints or [],
    )


class TestPromptContract(unittest.TestCase):
    def test_version_constant_is_stable(self):
        self.assertEqual(PLANNING_PROMPT_VERSION, "planning-v1")

    def test_requires_strict_json_output(self):
        prompt = _render()
        self.assertIn("只输出严格 JSON", prompt)

    def test_requires_direction_gap_and_phased_actions(self):
        prompt = _render()
        self.assertIn("target_roles", prompt)
        self.assertIn("gap_analysis", prompt)
        self.assertIn("roadmap_30_90_180", prompt)
        self.assertIn("30d", prompt)
        self.assertIn("90d", prompt)
        self.assertIn("180d", prompt)

    def test_freezes_complete_output_schema(self):
        # The lead freezes the output structure for evaluation; every
        # CareerPlanResponse field must appear in the prompt schema.
        prompt = _render()
        for field in (
            "user_facing_advice",
            "target_roles",
            "gap_analysis",
            "roadmap_30_90_180",
            "learning_resources",
            "next_actions",
            "risk_flags",
            "follow_up_questions",
            "confidence",
        ):
            self.assertIn(f'"{field}"', prompt, field)

    def test_injects_goal_profile_and_intent(self):
        prompt = _render()
        self.assertIn(GOAL, prompt)
        self.assertIn(TEXT, prompt)
        self.assertIn(INTENT, prompt)
        self.assertIn(PROFILE, prompt)

    def test_injects_knowledge_hints_as_citable_facts(self):
        hints = ["数据分析师 | 核心技能: SQL | 薪资参考: 10-20k"]
        prompt = _render(hints)
        self.assertIn("SQL", prompt)
        self.assertIn("数据分析师", prompt)

    def test_forbids_fabricating_career_facts(self):
        prompt = _render()
        self.assertIn("不得虚构", prompt)
        self.assertIn("引用\"知识库提示\"中的事实", prompt)

    def test_empty_knowledge_is_flagged_instead_of_silent(self):
        prompt = _render([])
        self.assertIn("知识库未命中", prompt)
        self.assertIn("risk_flags", prompt)

    def test_prompt_contains_no_unresolved_placeholders(self):
        prompt = _render(["示例岗位 | 技能: Python"])
        self.assertNotIn("{user_goal}", prompt)
        self.assertNotIn("{knowledge", prompt)


if __name__ == "__main__":
    unittest.main()
