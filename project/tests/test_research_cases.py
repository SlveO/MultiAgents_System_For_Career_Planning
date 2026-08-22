from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

from project.experiments.architecture_protocol import ResearchContracts
from project.experiments.run_architecture_experiments import load_cases


class TestResearchCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.contracts = ResearchContracts(cls.root)
        cls.cases = load_cases(contracts=cls.contracts)
        cls.source_leads_text = (
            cls.root / "dataset" / "career_source_leads.json"
        ).read_text(encoding="utf-8")
        cls.source_leads = json.loads(cls.source_leads_text)

    def test_pilot_has_three_images_and_three_pdf_pages(self) -> None:
        self.assertEqual(len(self.cases), 6)
        self.assertEqual(
            [case["modality"] for case in self.cases].count("image"), 3
        )
        self.assertEqual(
            [case["modality"] for case in self.cases].count("pdf_page"), 3
        )
        self.assertTrue(all(case["sets"] == ["pilot"] for case in self.cases))

    def test_cases_freeze_four_media_only_evidence_items(self) -> None:
        for case in self.cases:
            self.assertEqual(len(case["expected_evidence"]), 4)
            self.assertTrue(
                all(item["required_for_plan"] for item in case["expected_evidence"])
            )
            non_media_text = json.dumps(
                {
                    "user_goal": case["user_goal"],
                    "fixed_profile": case["fixed_profile"],
                    "knowledge_snippets": case["knowledge_snippets"],
                },
                ensure_ascii=False,
            )
            for expected in case["expected_evidence"]:
                self.assertNotIn(expected["fact"], non_media_text)

    def test_case_knowledge_ids_match_canonical_target_roles(self) -> None:
        knowledge = json.loads(
            (self.root / "dataset" / "career_knowledge_base.json").read_text(
                encoding="utf-8"
            )
        )
        roles_by_id = {
            f"career-{index:03d}": record["role"]
            for index, record in enumerate(knowledge, start=1)
        }
        for case in self.cases:
            target = case["fixed_profile"]["target_role"]
            for snippet in case["knowledge_snippets"]:
                self.assertEqual(roles_by_id[snippet["knowledge_id"]], target)

    def test_assets_are_anonymous_self_created_fixtures(self) -> None:
        for case in self.cases:
            self.assertEqual(case["source"]["source_type"], "self_created")
            self.assertIn("CC0-1.0", case["source"]["license"])
            self.assertTrue(case["privacy_reviewed"])
        combined = json.dumps(self.cases, ensure_ascii=False)
        self.assertNotRegex(combined, r"[A-Za-z]:\\Users\\")
        self.assertNotRegex(combined, r"\b1[3-9]\d{9}\b")
        self.assertNotRegex(combined, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def test_source_leads_are_not_mislabeled_as_ground_truth(self) -> None:
        self.assertEqual(
            self.source_leads["schema_version"], "career-source-leads-v1"
        )
        records = self.source_leads["records"]
        self.assertEqual(len(records), 20)
        self.assertEqual(len({record["role"] for record in records}), 20)
        self.assertIn("骨科医师", {record["role"] for record in records})
        self.assertNotIn("大骨科医师", {record["role"] for record in records})
        for record in records:
            review = record["review"]
            self.assertEqual(review["status"], "provenance_lead_only")
            self.assertEqual(review["verified_fields"], [])
            self.assertIn("member_claimed_supported_fields", review)
            parsed = urlparse(record["source_lead"]["url"])
            self.assertEqual(parsed.scheme, "https")
            self.assertTrue(parsed.netloc)
        self.assertNotIn("reference:7", self.source_leads_text)

    def test_source_leads_contain_no_private_paths_or_secrets(self) -> None:
        self.assertNotRegex(self.source_leads_text, r"[A-Za-z]:\\Users\\")
        self.assertNotIn("DEEPSEEK_API_KEY=", self.source_leads_text)
        self.assertNotRegex(
            self.source_leads_text,
            re.compile(r"(?:api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_-]{16,}", re.I),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
