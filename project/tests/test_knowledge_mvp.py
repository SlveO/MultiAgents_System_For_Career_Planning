from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.career_knowledge import CareerKnowledgeBase


class TestKnowledgeMvp(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[2] / "dataset" / "career_knowledge_base.json"

    def test_knowledge_base_has_at_least_sixty_five_stably_identified_roles(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))

        ids = [item.item_id for item in knowledge.items]
        roles = [item.role for item in knowledge.items]
        self.assertGreaterEqual(len(knowledge.items), 65)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(roles), len(set(roles)))
        self.assertTrue(all(item_id.startswith("career-") for item_id in ids))

    def test_keyword_retrieval_is_default_and_returns_ids(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))

        with patch.object(
            knowledge,
            "_init_embedder",
            side_effect=AssertionError("default retrieval must not initialize embeddings"),
        ):
            hits = knowledge.retrieve("Python SQL 数据分析", top_k=3)

        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(all(hit["item_id"].startswith("career-") for hit in hits))
        self.assertIn("数据分析师", [hit["role"] for hit in hits])

    def test_natural_chinese_query_prioritizes_matching_role(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))

        hits = knowledge.retrieve("我想获得数据分析实习", top_k=3)

        self.assertEqual(hits[0]["role"], "数据分析师")

    def test_integrated_member_sources_are_preserved_without_duplicate_roles(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))
        items = {item.role: item for item in knowledge.items}

        self.assertTrue(items["算法工程师"].sources)
        self.assertIn("统计学", items["算法工程师"].suitable_majors)
        self.assertTrue(items["高分子材料工程师"].sources)
        self.assertEqual(len(items), len(knowledge.items))

    def test_anonymized_retrieval_cases_return_the_target_role(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))
        cases_path = self.path.with_name("career_retrieval_cases.json")
        cases = json.loads(cases_path.read_text(encoding="utf-8"))

        for case in cases:
            with self.subTest(case_id=case["id"]):
                profile = case["profile"]
                query = " ".join(
                    [
                        profile["major"],
                        *profile["skills"],
                        *profile["interests"],
                        profile["target_role"],
                    ]
                )
                roles = [hit["role"] for hit in knowledge.retrieve(query, top_k=5)]
                self.assertIn(profile["target_role"], roles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
