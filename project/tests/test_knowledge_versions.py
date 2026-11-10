from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.career_knowledge import CareerKnowledgeBase
from project.core.knowledge_versions import KnowledgeConflict, KnowledgeRegistry, KnowledgeRegistryError


class TestKnowledgeVersions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seed_path = self.root / "seed.json"
        self.entries = [{
            "id": "career-001", "role": "数据分析师", "skills": ["SQL", "Python"],
            "resources": ["SQLBolt"], "transition_paths": ["运营 -> 数据分析"],
            "salary_hint": "历史参考", "sources": ["https://example.com/source"],
        }, {
            "id": "career-002", "role": "后端工程师", "skills": ["Java"],
            "resources": [], "transition_paths": [], "salary_hint": "", "sources": [],
        }]
        self.seed_path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")
        self.registry = KnowledgeRegistry(self.root / "registry", self.seed_path)

    def publish_changed(self, *, skill="统计学"):
        entries = copy.deepcopy(self.entries)
        entries[0]["skills"].append(skill)
        candidate = self.registry.create_candidate(entries, author="管理员", note="补充技能候选")
        self.registry.review(candidate["id"], reviewer="审核人", approve=True, note="核对结构，来源仍待核验")
        return self.registry.publish(candidate["id"], actor="管理员"), candidate

    def test_seed_has_fixed_unique_ids_and_reordering_preserves_identity(self):
        path = Path(__file__).resolve().parents[2] / "dataset" / "career_knowledge_base.json"
        entries = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 65)
        self.assertEqual(len({entry["id"] for entry in entries}), 65)
        self.assertEqual(entries[0]["id"], "career-001")
        self.assertEqual(entries[0]["role"], "算法工程师")
        self.assertEqual(entries[1]["id"], "career-002")
        self.seed_path.write_text(json.dumps(list(reversed(entries)), ensure_ascii=False), encoding="utf-8")
        knowledge = CareerKnowledgeBase(kb_path=str(self.seed_path))
        self.assertEqual(knowledge.items[-1].item_id, "career-001")

    def test_snapshot_uses_existing_keyword_ranker_and_never_models(self):
        expected = CareerKnowledgeBase(kb_path=str(self.seed_path)).retrieve("数据分析 SQL", 4)
        with patch.object(CareerKnowledgeBase, "_init_embedder", side_effect=AssertionError("no model")):
            snapshot = self.registry.snapshot("数据分析 SQL")
        self.assertEqual(snapshot["references"], expected)
        self.assertEqual(snapshot["hits"][0]["skills"], ["SQL", "Python"])
        self.assertEqual(snapshot["hits"][0]["subject"], "job")
        self.assertEqual(snapshot["hits"][0]["verification"], "unverified")
        self.assertEqual(self.registry.snapshot("nothing_matched")["hits"], [])

    def test_publication_requires_review_and_rejected_is_final(self):
        candidate = self.registry.create_candidate(self.entries, author="author")
        with self.assertRaises(KnowledgeConflict):
            self.registry.publish(candidate["id"], actor="author")
        self.registry.review(candidate["id"], reviewer="reviewer", approve=False, note="缺少依据")
        with self.assertRaises(KnowledgeConflict):
            self.registry.publish(candidate["id"], actor="author")
        with self.assertRaises(KnowledgeConflict):
            self.registry.review(candidate["id"], reviewer="reviewer", approve=True, note="更改决定")

    def test_snapshot_stays_identical_across_publication_rollback_and_restart(self):
        initial = self.registry.current()["version_id"]
        snapshot = self.registry.snapshot("数据分析")
        saved = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        released, candidate = self.publish_changed()
        self.assertNotEqual(initial, released["version_id"])
        self.assertIn("统计学", self.registry.snapshot("数据分析")["hits"][0]["skills"])
        self.assertEqual(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), saved)
        self.registry.rollback(initial, actor="管理员", note="试用回滚")
        self.assertEqual(self.registry.snapshot("数据分析"), snapshot)
        self.seed_path.unlink()
        restarted = KnowledgeRegistry(self.registry.root, self.seed_path)
        self.assertEqual(restarted.snapshot("数据分析"), snapshot)
        self.assertEqual(len(restarted.list_versions()), 2)
        self.assertEqual([event["action"] for event in restarted.release_events()], ["bootstrap", "publish", "rollback"])
        # Retrying an already published candidate cannot undo a later rollback.
        restarted.publish(candidate["id"], actor="管理员")
        self.assertEqual(restarted.current()["version_id"], initial)

    def test_stale_review_does_not_overwrite_a_new_release(self):
        stale = self.registry.create_candidate(self.entries, author="author")
        self.registry.review(stale["id"], reviewer="reviewer", approve=True, note="检查完成")
        released, _ = self.publish_changed()
        with self.assertRaises(KnowledgeConflict):
            self.registry.publish(stale["id"], actor="author")
        self.assertEqual(self.registry.current()["version_id"], released["version_id"])

    def test_candidate_diff_is_by_fixed_id_and_does_not_raise_source_trust(self):
        entries = copy.deepcopy(self.entries)
        entries[0]["skills"].append("统计学")
        candidate = self.registry.create_candidate(list(reversed(entries)), author="author")
        diff = self.registry.diff_candidate(candidate["id"])
        self.assertEqual(diff["added"], [])
        self.assertEqual(diff["removed"], [])
        self.assertEqual([entry["id"] for entry in diff["changed"]], ["career-001"])
        self.registry.review(candidate["id"], reviewer="reviewer", approve=True, note="允许发布而非事实核验")
        self.registry.publish(candidate["id"], actor="author")
        self.assertEqual(self.registry.snapshot("SQL")["hits"][0]["verification_status"], "not_verified")

    def test_release_integrity_and_database_failure_are_explicit(self):
        with sqlite3.connect(self.registry.db_path) as conn:
            conn.execute("UPDATE versions SET entries_json='[]'")
        with self.assertRaisesRegex(KnowledgeRegistryError, "integrity"):
            self.registry.snapshot("SQL")
        self.registry.db_path.unlink()
        self.registry.db_path.write_text("not a sqlite file", encoding="utf-8")
        with self.assertRaises(KnowledgeRegistryError):
            self.registry.current()

    def test_duplicate_or_missing_ids_and_fake_verified_metadata_are_rejected(self):
        for entries in [
            [self.entries[0], self.entries[0]],
            [{key: value for key, value in self.entries[0].items() if key != "id"}],
            [{**self.entries[0], "verification_status": "verified"}],
        ]:
            with self.assertRaises(ValueError):
                self.registry.create_candidate(entries, author="author")

    def test_content_checksum_covers_complete_entries(self):
        current = self.registry.current()
        raw = json.dumps(current["entries"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(current["content_sha256"], hashlib.sha256(raw.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
