import unittest

from project.core.output_adaptation import adapt_output, brief_reading_view, original_version
from project.core.output_preferences import OutputPreferences, render_initial_output
from project.core.schemas import CareerPlanResponse, UserProfile
from project.tests.test_output_adaptation import sample_response, FakeFeedbackModel


class TestReadingView(unittest.TestCase):
    def test_budget_selection_zero_unicode_and_transitions(self):
        response = sample_response(True)
        response.target_roles = ['😀岗位']
        response.profile.constraints.financial_budget_cny = 0
        response.output_version = original_version(response)
        initial = render_initial_output(response, OutputPreferences(detail='brief'))
        view = initial.reading_view
        self.assertLessEqual(len(view.excerpt), 500)
        self.assertIn('😀岗位', view.excerpt)
        self.assertIn('0小时', view.excerpt)
        self.assertIn('0元', view.excerpt)
        self.assertEqual(initial.display_text, response.user_facing_advice)
        model = FakeFeedbackModel(RuntimeError('must not be called'))
        repeat = adapt_output(response, '过于详细', 'a', model, baseline=initial)
        self.assertEqual((repeat.status, repeat.reason), ('unchanged', 'already_brief'))
        full = adapt_output(response, '过短', 'b', model, baseline=initial)
        self.assertEqual((full.reason, full.output_version.parent_version), ('full_view', 1))
        self.assertIsNone(full.output_version.reading_view)
        brief = adapt_output(response, '过于详细', 'c', model, baseline=full.output_version)
        self.assertEqual(brief.output_version.reading_view, view)
        self.assertFalse(model.calls)

    def test_oversized_required_and_prose_fallback(self):
        response = sample_response()
        response.target_roles = ['长' * 600]
        view = brief_reading_view(response, response.user_facing_advice)
        self.assertIn('长' * 600, view.excerpt)
        self.assertIn('超过', view.notice)
        response = CareerPlanResponse(session_id='s', intent='planning', profile=UserProfile(),
                                      plan_id='bare', user_facing_advice='文' * 800)
        view = brief_reading_view(response, response.user_facing_advice)
        self.assertEqual(view.excerpt, response.user_facing_advice)
        self.assertIn('无法安全摘选', view.notice)

    def test_deduplicates_only_identical_source_items(self):
        response = sample_response()
        response.next_actions = ['行动', '行动']
        view = brief_reading_view(response, response.user_facing_advice)
        self.assertEqual(view.excerpt.count('- 行动\n'), 1)
        self.assertIn('next_actions:0', view.item_ids)
        self.assertNotIn('next_actions:1', view.item_ids)


if __name__ == '__main__':
    unittest.main()
