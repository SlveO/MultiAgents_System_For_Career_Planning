from __future__ import annotations

import unittest

from pydantic import ValidationError

from project.core.output_adaptation import build_catalog, facts_hash, original_version
from project.core.output_preferences import OutputPreferences, preferences_prompt_hint, render_initial_output
from project.core.schemas import CareerPlanResponse, UserProfile
from project.tests.test_output_adaptation import sample_response


class TestOutputPreferenceContract(unittest.TestCase):
    def test_preferences_are_strict_user_reported_and_source_labelled(self):
        preferences = OutputPreferences(detail="brief", knowledge_level="beginner", source="user_input")
        self.assertTrue(preferences.user_reported)
        self.assertEqual(preferences.source, "user_input")
        for invalid in (
            {"detail": "tiny"}, {"knowledge_level": "expert"}, {"presentation": "chart"},
            {"source": "model_inference"}, {"user_reported": False}, {"user_reported": 1},
            {"user_reported": "true"}, {"detail": 1}, {"cognitive_load": 0.2},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                OutputPreferences.model_validate(invalid)

    def test_prompt_labels_preferences_without_claiming_knowledge_or_cognition(self):
        self.assertEqual(preferences_prompt_hint(None), "")
        for knowledge_level in ("unspecified", "beginner", "familiar"):
            hint = preferences_prompt_hint(OutputPreferences(knowledge_level=knowledge_level))
            self.assertIn("用户自报", hint)
            self.assertIn("不是已核实的能力或职业事实", hint)
            self.assertIn("不是认知负荷测量", hint)
            self.assertIn("资金预算", hint)
            self.assertIn("来源：cli", hint)
        self.assertIn("不证明其实际技能不足", preferences_prompt_hint(OutputPreferences(knowledge_level="beginner")))
        self.assertIn("不证明其已经掌握", preferences_prompt_hint(OutputPreferences(knowledge_level="familiar")))


class TestInitialOutputPreferences(unittest.TestCase):
    def test_missing_preferences_preserve_original_display(self):
        response = sample_response()
        original = response.model_dump()
        result = render_initial_output(response, None)
        expected = original_version(response)
        self.assertEqual(result.model_dump(exclude={"created_at"}), expected.model_dump(exclude={"created_at"}))
        self.assertEqual(response.model_dump(), original)

    def test_levels_retain_essential_source_items_and_increase_available_detail(self):
        response = sample_response()
        response.gap_analysis = [f"能力差距{index}" for index in range(6)]
        response.learning_resources = [f"学习资源{index}" for index in range(6)]
        response.output_version = original_version(response)
        original = response.model_dump()
        catalog = build_catalog(response)
        essential = {item.item_id for item in catalog if item.essential}
        available = {item.item_id for item in catalog}
        versions = {detail: render_initial_output(response, OutputPreferences(detail=detail))
                    for detail in ("brief", "standard", "detailed")}
        brief = versions.pop("brief")
        self.assertEqual(brief.display_text, response.user_facing_advice)
        self.assertLessEqual(len(brief.reading_view.excerpt), 500)
        for detail, version in versions.items():
            with self.subTest(detail=detail):
                self.assertTrue(essential <= set(version.item_ids))
                self.assertTrue(set(version.item_ids) <= available)
                self.assertEqual(version.version, 1)
                self.assertIsNone(version.parent_version)
                self.assertEqual(version.plan_id, response.plan_id)
                self.assertEqual(version.facts_sha256, facts_hash(response))
                self.assertEqual(version.source_text_sha256, response.output_version.source_text_sha256)
                self.assertIn("30天目标", version.display_text)
                self.assertIn("90天目标", version.display_text)
                self.assertIn("180天目标", version.display_text)
        self.assertEqual(set(versions["detailed"].item_ids), available)
        self.assertLess(len(versions["standard"].display_text), len(versions["detailed"].display_text))
        self.assertEqual(sum(item.startswith("gap_analysis:") for item in versions["standard"].item_ids), 3)
        self.assertEqual(sum(item.startswith("learning_resources:") for item in versions["standard"].item_ids), 3)
        self.assertEqual(response.model_dump(), original)

    def test_zero_budgets_unresolved_fields_and_private_information_are_preserved_safely(self):
        response = sample_response()
        response.profile.constraints.financial_budget_cny = 0
        response.profile.main_constraints.append("联系 demo@example.com 或13812345678")
        from project.core.schemas import FieldState
        response.profile.guidance.fields["skills"] = FieldState(status="refused")
        response.output_version = original_version(response)
        for detail in ("brief", "standard", "detailed"):
            with self.subTest(detail=detail):
                result = render_initial_output(response, OutputPreferences(detail=detail))
                if detail == "brief":
                    self.assertIn("每周时间：0小时", result.reading_view.excerpt)
                    self.assertIn("资金预算：0元", result.reading_view.excerpt)
                    continue
                self.assertIn("每周时间：0小时", result.display_text)
                self.assertIn("资金预算：0元", result.display_text)
                self.assertIn("技能：暂不回答", result.display_text)
                self.assertIn("[REDACTED_EMAIL]", result.display_text)
                self.assertIn("[REDACTED_PHONE]", result.display_text)
                self.assertNotIn("demo@example.com", result.display_text)
                self.assertNotIn("13812345678", result.display_text)

    def test_paragraph_preference_changes_layout_without_changing_source_plan(self):
        response = sample_response()
        original = response.model_dump()
        bullets = render_initial_output(response, OutputPreferences(detail="standard", presentation="bullets"))
        paragraphs = render_initial_output(response, OutputPreferences(detail="standard", presentation="paragraph"))
        self.assertEqual(bullets.item_ids, paragraphs.item_ids)
        self.assertIn("- 数据分析师", bullets.display_text)
        self.assertNotIn("- 数据分析师", paragraphs.display_text)
        self.assertIn("数据分析师", paragraphs.display_text)
        self.assertEqual(response.model_dump(), original)

    def test_timeline_is_chronological_without_reordering_source_response(self):
        response = sample_response()
        response.roadmap_30_90_180.reverse()
        original = response.model_dump()
        result = render_initial_output(response, OutputPreferences(detail="detailed"))
        self.assertLess(result.display_text.index("30天目标"), result.display_text.index("90天目标"))
        self.assertLess(result.display_text.index("90天目标"), result.display_text.index("180天目标"))
        self.assertEqual(response.model_dump(), original)

    def test_preferences_never_become_factual_profile_information(self):
        response = sample_response()
        original = response.model_dump()
        beginner = render_initial_output(response, OutputPreferences(knowledge_level="beginner"))
        familiar = render_initial_output(response, OutputPreferences(knowledge_level="familiar"))
        self.assertEqual(beginner.display_text, familiar.display_text)
        self.assertEqual(beginner.facts_sha256, familiar.facts_sha256)
        self.assertEqual(response.model_dump(), original)
        self.assertNotIn("beginner", response.model_dump_json())

    def test_sparse_prose_uses_existing_text_and_empty_plan_does_not_invent_content(self):
        response = CareerPlanResponse(session_id="demo", intent="planning", plan_id="sparse",
                                      profile=UserProfile(), user_facing_advice="先澄清你的职业目标。")
        result = render_initial_output(response, OutputPreferences(detail="brief"))
        self.assertIn(response.user_facing_advice, result.display_text)
        self.assertEqual(result.reading_view.excerpt, response.user_facing_advice)
        self.assertIn("无法安全摘选", result.reading_view.notice)
        response.user_facing_advice = ""
        empty = render_initial_output(response, OutputPreferences(detail="brief"))
        self.assertEqual(empty.display_text, "")
        self.assertEqual(empty.item_ids, [])


if __name__ == "__main__":
    unittest.main()
