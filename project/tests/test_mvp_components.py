import unittest
import uuid
from pathlib import Path

from project.core.schemas import TaskRequest
from project.core.career_knowledge import CareerKnowledgeBase
from project.core.session_memory import SessionMemory


class TestMVPComponents(unittest.TestCase):
    @staticmethod
    def _tmp_dir(name: str) -> Path:
        p = Path("project/data/test_tmp") / f"{name}_{uuid.uuid4().hex[:8]}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def test_schema_build(self):
        req = TaskRequest(session_id="s1", user_goal="我想转行做数据分析")
        self.assertEqual(req.session_id, "s1")
        self.assertEqual(req.user_goal, "我想转行做数据分析")

    def test_memory_crud(self):
        d = self._tmp_dir("memory")
        db = SessionMemory(db_path=str(d / "mem.db"))
        db.upsert_profile("s1", {"interests": ["数据分析"]})
        profile = db.get_profile("s1")
        self.assertIn("interests", profile)
        db.append_feedback("s1", "建议很实用", 5)
        db.append_interaction("s1", {"q": 1}, {"a": 2})
        history = db.get_session_history("s1")
        self.assertTrue(len(history) >= 1)

    def test_knowledge_search(self):
        d = self._tmp_dir("knowledge")
        kb = CareerKnowledgeBase(kb_path=str(d / "kb.json"))
        hits = kb.search("我想做嵌入式和C++开发", top_k=2)
        self.assertTrue(len(hits) >= 1)
        hints = kb.to_hints("我想做数据分析", top_k=2)
        self.assertTrue(len(hints) >= 1)


if __name__ == "__main__":
    unittest.main()
