from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.career_knowledge import CareerKnowledgeBase


class TestKnowledgeMvp(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[2] / "dataset" / "career_knowledge_base.json"

    def test_knowledge_base_has_at_least_fifty_stably_identified_roles(self) -> None:
        knowledge = CareerKnowledgeBase(kb_path=str(self.path))

        ids = [item.item_id for item in knowledge.items]
        roles = [item.role for item in knowledge.items]
        self.assertGreaterEqual(len(knowledge.items), 50)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
