from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from project.core.local_guidance import parse_guidance_response
from project.core.schemas import GuidanceDecision


class TestGuidancePlaceholders(unittest.TestCase):
    def fixture(self):
        return {
            'observations': [
                {'field': 'skills', 'value': 'Python', 'evidence_ids': ['source-1']},
                {'field': 'target_role', 'value': '', 'evidence_ids': []},
            ],
            'question': {'field': 'target_role', 'reason': 'missing', 'prompt': '想尝试哪个具体岗位？', 'evidence_ids': []},
        }

    def test_mixed_empty_placeholder_is_removed_without_rewriting_valid_information(self):
        value = self.fixture()
        with self.assertRaises(ValidationError):
            GuidanceDecision.model_validate(value)
        decision, dropped = parse_guidance_response(json.dumps(value, ensure_ascii=False))
        self.assertEqual(dropped, 1)
        self.assertEqual(decision.observations[0].model_dump(), value['observations'][0])
        self.assertEqual(decision.question.field, 'target_role')

    def test_all_invalid_or_nonempty_invalid_observations_still_fail(self):
        invalid = [
            {'field': 'education_stage', 'value': '本科大三', 'evidence_ids': ['source-1']},
            {'field': 'target_role', 'value': '', 'evidence_ids': []},
            {'field': 'target_role', 'value': '不知道', 'evidence_ids': []},
            {'field': 'invented_field', 'value': '', 'evidence_ids': []},
        ]
        for observation in invalid:
            with self.subTest(observation=observation):
                value = self.fixture()
                value['observations'] = [observation]
                with self.assertRaises(ValueError):
                    parse_guidance_response(json.dumps(value))

    def test_invalid_question_extra_keys_and_excess_observations_are_not_normalized(self):
        for change in ['question', 'extra', 'too_many', 'observation_extra']:
            with self.subTest(change=change):
                value = self.fixture()
                if change == 'question':
                    value['question']['field'] = 'invented_field'
                elif change == 'extra':
                    value['action'] = 'invented_action'
                elif change == 'too_many':
                    value['observations'] *= 13
                else:
                    value['observations'][1]['hidden'] = 'not allowed'
                with self.assertRaises(ValueError):
                    parse_guidance_response(json.dumps(value))

    def test_declared_empty_observation_list_remains_valid(self):
        decision, dropped = parse_guidance_response('{"observations":[],"question":null}')
        self.assertEqual(dropped, 0)
        self.assertEqual(decision.observations, [])


if __name__ == '__main__':
    unittest.main()
