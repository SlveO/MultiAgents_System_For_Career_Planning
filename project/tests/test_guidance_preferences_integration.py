from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from project import assistant_cli
from project.core.guidance import run_guidance
from project.core.coaching_policy import build_coaching_context
from project.core.intake import apply_answers_to_profile
from project.core.output_adaptation import facts_hash
from project.core.output_preferences import OutputPreferences
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import TaskRequest, UserProfile
from project.orchestrator import CareerOrchestrator
from project.tests.test_guidance import FakeGuidanceModel
from project.tests.test_output_adaptation import AdaptableBrain, FakeFeedbackModel


class TestGuidancePreferencesIntegration(unittest.TestCase):
    def make_orchestrator(self, root):
        return CareerOrchestrator(
            db_path=str(root / 'sessions.db'), brain_client=AdaptableBrain(long=True),
            guidance_model=FakeGuidanceModel(), feedback_model=FakeFeedbackModel(),
            run_logger=JsonlRunLogger(root / 'runs.jsonl'),
        )

    def test_explicit_brief_preferences_reach_prompt_display_storage_and_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orch = self.make_orchestrator(root)
            original_plan = orch.cloud_brain.plan
            prompts = []
            def plan(prompt, model=None):
                prompts.append(prompt)
                return original_plan(prompt, model)
            orch.cloud_brain.plan = plan
            output = []
            inputs = iter(['1', '2'])
            code = assistant_cli.main(
                ['--goal', '求职', '--no-follow-up', '--financial-budget', '0', '--detail', 'brief',
                 '--knowledge-level', 'beginner', '--presentation', 'bullets', '--feedback-rounds', '2'],
                input_fn=lambda _: next(inputs), output_fn=output.append, orchestrator_factory=lambda: orch,
            )
            self.assertEqual(code, 0)
            saved = orch.get_saved_plan('default-session')
            history = orch.memory.get_plan_interaction('default-session', saved.plan_id)
            self.assertEqual(history['request']['output_preferences']['detail'], 'brief')
            self.assertEqual(history['request']['output_preferences']['knowledge_level'], 'beginner')
            self.assertIn('不证明其实际技能不足', prompts[0])
            versions = orch.memory.get_output_versions('default-session', saved.plan_id)
            self.assertEqual([version['version'] for version in versions], [1, 2])
            self.assertEqual(versions[0]['display_text'], saved.user_facing_advice)
            excerpt = versions[0]['reading_view']['excerpt']
            self.assertLess(len(excerpt), len(saved.user_facing_advice))
            self.assertIn(excerpt, output)
            self.assertIn('0元', excerpt)
            self.assertIsNone(versions[1]['reading_view'])
            self.assertIn(saved.user_facing_advice, output)
            self.assertTrue(all(v['facts_sha256'] == facts_hash(saved) for v in versions))
            self.assertEqual(saved.profile.skills, [])
            restarted = self.make_orchestrator(root)
            kept = restarted.adapt_feedback('default-session', '合适', request_id='restart')
            self.assertEqual(kept.source_version, 2)
            self.assertFalse(restarted.feedback_model.calls)

    def test_no_preference_defaults_preserve_original_and_are_not_user_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            orch = self.make_orchestrator(Path(tmp))
            response = orch.run(TaskRequest(session_id='s', user_goal='求职'))
            self.assertEqual(response.output_version.display_text, response.user_facing_advice)
            saved = orch.memory.get_plan_interaction('s', response.plan_id)
            self.assertIsNone(saved['request']['output_preferences'])

    def test_resume_rejects_initial_preference_override_before_creating_orchestrator(self):
        factory = MagicMock()
        code = assistant_cli.main(['--resume-feedback', '--detail', 'brief'],
                                  orchestrator_factory=factory, output_fn=lambda _: None)
        self.assertEqual(code, 2)
        factory.assert_not_called()

    def test_policy_is_sent_to_model_and_trace_is_persistable(self):
        model = FakeGuidanceModel()
        req = TaskRequest(session_id='s', user_goal='求职')
        answers = iter(['后端开发工程师', 'Python'])
        result = run_guidance(req, UserProfile(), [], model, max_rounds=2,
                              input_fn=lambda _: next(answers), output_fn=lambda _: None)
        self.assertIn('coaching_policy', model.contexts[0])
        self.assertNotIn('candidates', model.contexts[0]['coaching_policy'])
        self.assertNotIn('action_rules', model.contexts[0]['coaching_policy'])
        self.assertNotIn('question', model.contexts[0]['coaching_policy'])
        trace = json.loads(req.metadata['coaching_trace'])
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]['focus_field'], 'target_role')
        self.assertGreater(trace[0]['metrics']['follow_up_threshold'], trace[1]['metrics']['follow_up_threshold'])
        self.assertEqual(result.guidance.fields['target_role'].status, 'known')
        self.assertEqual(result.guidance.fields['skills'].status, 'known')

    def test_logged_policy_focus_matches_the_actual_question_with_multiple_conflicts(self):
        profile = apply_answers_to_profile(UserProfile(), {
            'education': '本科', 'major': '计算机', 'skills': 'Python', 'interests': '开发',
            'target_role': '后端开发工程师', 'time_budget': '8小时', 'preference': '上海', 'constraints': '无',
        })
        for field in ['interests', 'preference']:
            profile.guidance.fields[field].status = 'conflict'
            profile.guidance.fields[field].values = ['选项甲', '选项乙']
        expected = build_coaching_context(profile, round_number=1, max_rounds=1)
        req = TaskRequest(session_id='s', user_goal='求职')
        prompts = []
        run_guidance(req, profile, [], FakeGuidanceModel(), max_rounds=1,
                     input_fn=lambda prompt: prompts.append(prompt) or '结束引导', output_fn=lambda _: None)
        trace = json.loads(req.metadata['coaching_trace'])
        self.assertEqual(trace[0]['focus_field'], expected['focus_field'])
        self.assertIn(expected['question']['prompt'], prompts[0])

    def test_preferences_are_strict_and_do_not_add_factual_profile_fields(self):
        req = TaskRequest(session_id='s', user_goal='求职',
                          output_preferences={'detail': 'brief', 'knowledge_level': 'familiar'})
        self.assertIsInstance(req.output_preferences, OutputPreferences)
        self.assertEqual(UserProfile().skills, [])
        self.assertNotIn('knowledge_level', UserProfile.model_fields)


if __name__ == '__main__':
    unittest.main()
