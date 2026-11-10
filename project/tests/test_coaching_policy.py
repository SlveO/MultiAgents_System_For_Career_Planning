from __future__ import annotations

import unittest

from project.core.coaching_policy import build_coaching_context
from project.core.guidance import question_key
from project.core.intake import FIELD_PATHS, apply_answers_to_profile, initialize_fields, record_answer
from project.core.schemas import FieldState, GuidanceEvidence, GuidanceQuestion, UserProfile


def complete_profile():
    return apply_answers_to_profile(UserProfile(), {
        'education': '本科大三', 'major': '统计学', 'skills': 'Python、SQL',
        'interests': '分析业务问题', 'target_role': '数据分析师', 'time_budget': '8',
        'preference': '上海', 'constraints': '无',
    })


class TestCoachingPolicy(unittest.TestCase):
    def test_missing_goal_is_first_and_question_matches_existing_contract(self):
        profile = UserProfile()
        initialize_fields(profile)
        before = profile.model_dump()
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'clarify_goal')
        self.assertEqual(context['focus_field'], 'target_role')
        self.assertEqual(context['question']['reason'], 'missing')
        question = GuidanceQuestion.model_validate(context['question'])
        selected = next(item for item in context['candidates'] if item['focus_field'] == 'target_role')
        self.assertEqual(selected['question_key'], question_key(profile, question))
        self.assertEqual(context['metrics']['information_completeness'], 0)
        self.assertEqual(context['metrics']['evidence_coverage'], 0)
        self.assertEqual(profile.model_dump(), before)

    def test_conflict_precedes_core_missing_or_optional_questions(self):
        profile = complete_profile()
        evidence = profile.guidance.fields['major'].evidence_ids
        profile.guidance.fields['major'] = FieldState(status='conflict', values=['统计学', '数学'], evidence_ids=evidence)
        profile.guidance.fields['target_role'] = FieldState(status='missing')
        context = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertEqual(context['action'], 'resolve_conflict')
        self.assertEqual(context['focus_field'], 'major')
        self.assertEqual(context['selection_reason'], 'unresolved_conflict')
        self.assertEqual(context['question']['reason'], 'conflict')
        selected = next(item for item in context['candidates'] if item['focus_field'] == 'major')
        self.assertEqual(selected['question_key'], question_key(profile, GuidanceQuestion.model_validate(context['question'])))

    def test_unknown_refused_and_explicit_none_are_never_queried(self):
        profile = apply_answers_to_profile(UserProfile(), {
            'target_role': '不知道', 'skills': '无', 'time_budget': '不想回答',
            'education': '本科', 'major': '数学', 'interests': '暂无',
            'preference': '拒答', 'constraints': '不知道',
        })
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'stop')
        self.assertEqual(context['stop_reason'], 'no_eligible_question')
        self.assertEqual(context['candidates'], [])
        self.assertEqual(context['metrics']['response_coverage'], 1)
        self.assertLess(context['metrics']['information_completeness'], 1)
        self.assertIn('target_role', context['unresolved_fields'])
        self.assertIn('skills', context['protected_fields'])

    def test_refusal_during_conflict_suppresses_further_clarification(self):
        profile = complete_profile()
        profile.guidance.fields['target_role'] = FieldState(status='conflict', values=['数据分析师', '后端开发工程师'])
        record_answer(profile, 'target_role', '不想回答', question_id='conflict-once', origin='interactive')
        self.assertEqual(profile.guidance.fields['target_role'].status, 'conflict')
        context = build_coaching_context(profile, round_number=2, max_rounds=4)
        conflict = next(item for item in context['candidates'] if item['focus_field'] == 'target_role')
        self.assertFalse(conflict['eligible'])
        self.assertEqual(conflict['blocked_reason'], 'user_boundary')
        self.assertIsNone(context['question'])

    def test_complete_profile_stops_without_extra_questions(self):
        profile = complete_profile()
        before = profile.model_dump()
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'stop')
        self.assertEqual(context['stop_reason'], 'complete')
        self.assertEqual(context['metrics']['information_completeness'], 1)
        self.assertEqual(context['metrics']['evidence_coverage'], 1)
        self.assertEqual(profile.model_dump(), before)

    def test_last_round_threshold_changes_readiness_but_not_facts(self):
        profile = complete_profile()
        profile.guidance.fields['interests'] = FieldState(status='missing')
        profile.interests = []
        profile.guidance.answers = [answer for answer in profile.guidance.answers if answer.field != 'interests']
        before = profile.model_dump()
        early = build_coaching_context(profile, round_number=1, max_rounds=4)
        late = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertEqual(early['focus_field'], 'interests')
        self.assertEqual(late['action'], 'stop')
        self.assertEqual(late['stop_reason'], 'sufficient_information')
        self.assertEqual(early['metrics']['information_completeness'], late['metrics']['information_completeness'])
        self.assertEqual(early['metrics']['evidence_coverage'], late['metrics']['evidence_coverage'])
        self.assertGreater(early['metrics']['follow_up_threshold'], late['metrics']['follow_up_threshold'])
        self.assertEqual(profile.model_dump(), before)

    def test_core_gap_remains_actionable_at_last_round(self):
        profile = complete_profile()
        profile.guidance.fields['time_budget'] = FieldState(status='missing')
        profile.constraints.time_budget_hours_per_week = None
        profile.guidance.answers = [answer for answer in profile.guidance.answers if answer.field != 'time_budget']
        context = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertGreater(context['metrics']['information_completeness'], context['metrics']['follow_up_threshold'])
        self.assertEqual(context['focus_field'], 'time_budget')
        self.assertEqual(context['selection_reason'], 'essential_information')

    def test_round_limits_stop_even_with_conflicts(self):
        profile = complete_profile()
        profile.guidance.fields['target_role'] = FieldState(status='conflict', values=['数据分析师', '后端开发'])
        for round_number, max_rounds in [(1, 0), (5, 4), (2, 1)]:
            with self.subTest(round_number=round_number, max_rounds=max_rounds):
                context = build_coaching_context(profile, round_number=round_number, max_rounds=max_rounds)
                self.assertEqual(context['stop_reason'], 'round_limit')
                self.assertIsNone(context['question'])
                self.assertEqual(context['round']['remaining_including_current'], 0)
        for round_number, max_rounds in [(0, 4), (1, -1)]:
            with self.assertRaises(ValueError):
                build_coaching_context(profile, round_number=round_number, max_rounds=max_rounds)

    def test_asked_key_and_empty_answer_do_not_reopen_same_question(self):
        profile = UserProfile()
        initialize_fields(profile)
        first = build_coaching_context(profile, round_number=1, max_rounds=4)
        candidate = next(item for item in first['candidates'] if item['focus_field'] == 'target_role')
        profile.guidance.asked_keys.append(candidate['question_key'])
        second = build_coaching_context(profile, round_number=2, max_rounds=4)
        self.assertNotEqual(second['focus_field'], 'target_role')
        self.assertEqual(next(item for item in second['candidates'] if item['focus_field'] == 'target_role')['blocked_reason'], 'already_asked')
        record_answer(profile, 'skills', '', question_id='skills-empty', origin='interactive')
        third = build_coaching_context(profile, round_number=3, max_rounds=4)
        self.assertNotEqual(third['focus_field'], 'skills')

    def test_specific_goals_are_prioritized_without_psychological_inference(self):
        profile = complete_profile()
        record_answer(profile, 'target_role', '数据分析师、后端开发工程师', question_id='provided', origin='answers_json')
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'prioritize')
        self.assertEqual(context['question']['reason'], 'clarify')
        self.assertEqual(context['focus_field'], 'target_role')
        self.assertEqual(context['policy_kind'], 'deterministic_rules')
        self.assertEqual({rule['action'] for rule in context['action_rules']}, {'resolve_conflict', 'clarify_goal', 'clarify_values', 'prioritize', 'collect_background'})
        self.assertNotIn('emotion', context['metrics'])

    def test_interactive_clarification_is_not_repeated(self):
        profile = complete_profile()
        record_answer(profile, 'target_role', '数据分析师、后端开发工程师', question_id='answered', origin='interactive')
        context = build_coaching_context(profile, round_number=2, max_rounds=4)
        self.assertIsNone(context['question'])
        candidate = next(item for item in context['candidates'] if item['focus_field'] == 'target_role')
        self.assertEqual(candidate['blocked_reason'], 'already_clarified')

    def test_complete_profile_still_prioritizes_unranked_preferences_once(self):
        profile = complete_profile()
        record_answer(profile, 'preference', '上海、杭州', question_id='provided', origin='answers_json')
        context = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertEqual(context['metrics']['information_completeness'], 1)
        self.assertEqual(context['action'], 'prioritize')
        self.assertEqual(context['focus_field'], 'preference')
        self.assertEqual(context['selection_reason'], 'unranked_preferences')
        record_answer(profile, 'preference', '上海优先、杭州也可', question_id='ranked', origin='interactive')
        protected = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertEqual(protected['stop_reason'], 'complete')
        self.assertIsNone(protected['question'])
        record_answer(profile, 'preference', '不想回答', question_id='declined', origin='interactive')
        declined = build_coaching_context(profile, round_number=4, max_rounds=4)
        self.assertIsNone(declined['question'])
        self.assertIn('preference', declined['protected_fields'])

    def test_original_evidence_is_prioritized_before_question_reference_limit(self):
        profile = complete_profile()
        record_answer(profile, 'target_role', '找工作', question_id='vague', origin='answers_json')
        original_id = profile.guidance.fields['target_role'].evidence_ids[0]
        stored_ids = [f'copied-{number}' for number in range(8)]
        profile.guidance.evidence.extend(GuidanceEvidence(
            evidence_id=evidence_id, origin='stored_profile:target_role', excerpt='找工作',
        ) for evidence_id in stored_ids)
        profile.guidance.fields['target_role'].evidence_ids = stored_ids + [original_id]
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['metrics']['evidence_coverage'], 1)
        self.assertEqual(context['action'], 'clarify_goal')
        self.assertEqual(context['question']['evidence_ids'][0], original_id)
        self.assertEqual(len(context['question']['evidence_ids']), 8)
        selected = next(item for item in context['candidates'] if item['focus_field'] == 'target_role')
        self.assertTrue(selected['eligible'])
        self.assertEqual(selected['question_key'], question_key(profile, GuidanceQuestion.model_validate(context['question'])))

    def test_stored_profile_is_complete_but_not_original_evidence(self):
        profile = complete_profile()
        profile.guidance.evidence = [item.model_copy(update={'origin': 'stored_profile:copied'}) for item in profile.guidance.evidence]
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['metrics']['information_completeness'], 1)
        self.assertEqual(context['metrics']['evidence_coverage'], 0)
        self.assertEqual(context['stop_reason'], 'complete')

    def test_vague_goal_and_missing_values_use_distinct_actions(self):
        profile = complete_profile()
        record_answer(profile, 'target_role', '找工作', question_id='vague', origin='answers_json')
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'clarify_goal')
        profile = complete_profile()
        for field in ['preference', 'interests']:
            profile.guidance.fields[field] = FieldState(status='missing')
        profile.guidance.answers = [item for item in profile.guidance.answers if item.field not in {'preference', 'interests'}]
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['action'], 'clarify_values')
        self.assertEqual(context['focus_field'], 'preference')

    def test_c_cpp_is_not_misread_as_multiple_career_options(self):
        profile = complete_profile()
        record_answer(profile, 'target_role', 'C/C++开发工程师', question_id='specific', origin='answers_json')
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['stop_reason'], 'complete')
        self.assertNotIn('prioritize', [item['action'] for item in context['candidates']])

    def test_missing_guidance_fields_are_read_without_synthetic_evidence(self):
        profile = UserProfile(skills=['SQL'], target_role='数据分析师')
        before = profile.model_dump()
        context = build_coaching_context(profile, round_number=1, max_rounds=4)
        self.assertEqual(context['focus_field'], 'time_budget')
        self.assertEqual(context['metrics']['evidence_coverage'], 0)
        self.assertEqual(profile.model_dump(), before)
        self.assertEqual(set(context['unresolved_fields']), set(FIELD_PATHS) - {'target_role', 'skills'})


if __name__ == '__main__':
    unittest.main()
