from __future__ import annotations

import unittest

from project.core.source_protocol import SourceClaim, claim_from_excerpt, confirm_claim, student_claims
from project.core.evaluation import EvaluationSubmission, make_evaluation, summarize_evaluations


class TestSourceProtocol(unittest.TestCase):
    def test_job_requirement_can_never_enter_student_capabilities(self):
        job = claim_from_excerpt("skills", "Python", "岗位要求掌握 Python", subject="job", source_id="jd-1")
        self.assertEqual(job.verification, "unverified")
        confirmed = confirm_claim(job)
        self.assertEqual(confirmed.subject, "job")
        self.assertEqual(confirmed.verification, "user_confirmed")
        self.assertEqual(student_claims([confirmed]), [])
        self.assertEqual(student_claims([job], require_confirmation=False), [])

    def test_unknown_and_reference_subjects_are_not_student_claims(self):
        for subject in ("unknown", "reference"):
            claim = SourceClaim(field="skills", value="SQL", subject=subject, verification="user_confirmed")
            self.assertEqual(student_claims([claim]), [])

    def test_student_confirmation_does_not_claim_independent_verification(self):
        student = claim_from_excerpt("skills", "SQL", "我掌握SQL", subject="student")
        self.assertEqual(student_claims([student]), [])
        confirmed = confirm_claim(student)
        self.assertEqual(confirmed.verification, "user_confirmed")
        self.assertEqual(student.verification, "unverified")
        self.assertEqual(student_claims([confirmed])[0].value, "SQL")
        self.assertEqual(confirmed.claim_id, student.claim_id)

    def test_source_verification_requires_its_reference_and_verifier(self):
        with self.assertRaises(ValueError):
            SourceClaim(field="skills", value="SQL", verification="source_verified")
        verified = SourceClaim(
            field="skills", value="SQL", subject="student", verification="source_verified",
            verification_reference="证书原件核验记录#1", verified_by="审核人",
        )
        self.assertEqual(confirm_claim(verified).verification, "source_verified")

    def test_excerpt_must_support_extracted_value(self):
        with self.assertRaises(ValueError):
            claim_from_excerpt("skills", "Java", "我掌握SQL", subject="student")


class TestHumanEvaluation(unittest.TestCase):
    def test_no_default_or_fabricated_score(self):
        record = make_evaluation({"plan_id": "plan-1", "output_version": 2, "comment": "稍后评价"})
        self.assertIsNone(record["scores"])
        summary = summarize_evaluations([record])
        self.assertEqual(summary["evaluations"], 1)
        for metric in summary["metrics"].values():
            self.assertEqual(metric, {"n": 0, "mean": None})

    def test_partial_scores_are_tied_to_the_exact_version(self):
        record = make_evaluation({"plan_id": "plan-1", "output_version": 3, "scores": {"clarity": 4}})
        self.assertEqual((record["plan_id"], record["output_version"]), ("plan-1", 3))
        summary = summarize_evaluations([record])
        self.assertEqual(summary["metrics"]["clarity"], {"n": 1, "mean": 4})
        self.assertEqual(summary["metrics"]["usefulness"], {"n": 0, "mean": None})

    def test_ratings_reject_boolean_string_float_and_out_of_range_values(self):
        for value in (True, "4", 4.0, 0, 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EvaluationSubmission.model_validate({"plan_id": "p", "output_version": 1, "scores": {"clarity": value}})
        for version in (0, True, "2", 2.0):
            with self.assertRaises(ValueError):
                EvaluationSubmission.model_validate({"plan_id": "p", "output_version": version})


if __name__ == "__main__":
    unittest.main()
