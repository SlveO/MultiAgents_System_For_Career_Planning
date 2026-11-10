from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from project.web.queue import ProcessLock, TaskRunner
from project.web.storage import ApiError, WebStore, fingerprint


class FakeEngine:
    def __init__(self):
        self.calls = []

    def process(self, kind, state, payload, files):
        self.calls.append(kind)
        state = json.loads(json.dumps(state))
        if kind == "perceive":
            state.update(extracted=True, status="awaiting_answer", question={"id": "q1", "field": "skills", "prompt": "你掌握什么技能？", "reason": "missing"})
        elif kind == "answers":
            state.update(status="ready", question=None)
        elif kind == "plans":
            version = {"plan_id": "plan1", "version": 1, "display_text": "匿名测试规划", "facts_sha256": "facts", "source_text_sha256": "source", "created_at": "2026-11-06T00:00:00Z"}
            state.update(status="planned", plan={"plan_id": "plan1", "report_markdown": "# 保存的报告\n", "knowledge_snapshot": {"version_id": "k1"}}, versions=[version], knowledge_snapshot={"version_id": "k1"})
        elif kind == "feedback":
            state["versions"].append({**state["versions"][-1], "version": len(state["versions"])+1, "display_text": "调整后的表达"})
        return state


class WebStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.now = 1800000000.0
        self.store = WebStore(self.root/"web.db", self.root/"uploads", clock=lambda: self.now)
        self.store.initialize()
        self.engine = FakeEngine()
        self.runner = TaskRunner(self.store, lambda: self.engine)
        self.counter = 0

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, owner="alice", key=None):
        self.counter += 1
        payload = {"user_goal": "匿名职业规划", "text_input": "本科学生"}
        return self.store.create_session(owner, key or str(self.counter), fingerprint("POST", "/v1/sessions", payload), payload)[0]

    def submit(self, session, kind, payload=None, owner="alice", key=None):
        self.counter += 1
        data = {"state_version": self.store.get_session(owner, session["id"])["state_version"], **(payload or {})}
        return self.store.enqueue(owner, session["id"], key or str(self.counter), fingerprint("POST", kind, data), kind, data)[0]

    def ready(self):
        session = self.create()
        self.submit(session, "perceive")
        self.runner.process_once()
        self.submit(session, "answers", {"question_id": "q1", "answer": "Python"})
        self.runner.process_once()
        return session

    def planned(self):
        session = self.ready()
        self.submit(session, "plans")
        self.runner.process_once()
        return session

    def test_create_idempotency_owner_scope_and_mismatch(self):
        first = self.create(key="same")
        again = self.create(key="same")
        self.assertEqual(first, again)
        self.assertNotEqual(first["id"], self.create("bob", key="same")["id"])
        with self.assertRaises(ApiError) as exc:
            self.store.create_session("alice", "same", "different", {"user_goal": "别的请求"})
        self.assertEqual(exc.exception.status, 409)

    def test_two_owners_cannot_read_delete_enqueue_or_get_tasks(self):
        session = self.create()
        task = self.submit(session, "perceive")
        for action in (
            lambda: self.store.get_session("bob", session["id"]),
            lambda: self.store.delete_session("bob", session["id"], "del", "del"),
            lambda: self.store.enqueue("bob", session["id"], "enq", "enq", "perceive", {"state_version": 1}),
            lambda: self.store.get_task("bob", task["id"]),
        ):
            with self.assertRaises(ApiError) as exc:
                action()
            self.assertEqual(exc.exception.status, 404)
        self.assertEqual(self.store.list_sessions("bob"), {"sessions": []})

    def test_upload_limits_type_and_private_paths(self):
        session = self.create()
        for index in range(3):
            self.store.add_upload("alice", session["id"], f"file{index}", f"file{index}", "test.txt", b"anonymous", "student")
        with self.assertRaises(ApiError) as exc:
            self.store.add_upload("alice", session["id"], "f4", "f4", "four.txt", b"four", "unknown")
        self.assertEqual(exc.exception.status, 413)
        self.assertNotIn("path", json.dumps(self.store.get_session("alice", session["id"])))
        session2 = self.create()
        with self.assertRaises(ApiError) as exc:
            self.store.add_upload("alice", session2["id"], "audio", "audio", "voice.wav", b"no", "unknown")
        self.assertEqual(exc.exception.status, 415)
        with self.assertRaises(ApiError):
            self.store.add_upload("alice", session2["id"], "large", "large", "test.txt", b"x"*(WebStore.MAX_FILE_BYTES+1), "unknown")

    def test_upload_replay_does_not_consume_slots_or_write_twice(self):
        session = self.create()
        args = ("alice", session["id"], "file", "file", "a.txt", b"anonymous", "job")
        self.assertEqual(self.store.add_upload(*args), self.store.add_upload(*args))
        self.assertEqual(len(list(self.store.upload_dir.iterdir())), 1)
        self.assertEqual(len(self.store.get_session("alice", session["id"])["uploads"]), 1)

    def test_duplicate_task_and_refreshed_question_guard(self):
        session = self.create()
        args = ("alice", session["id"], "perceive", "same", "perceive", {"state_version": 1})
        self.assertEqual(self.store.enqueue(*args), self.store.enqueue(*args))
        self.runner.process_once()
        restored = self.store.get_session("alice", session["id"])
        self.assertEqual(restored["question"]["id"], "q1")
        self.assertEqual(self.engine.calls, ["perceive"])
        with self.assertRaises(ApiError) as exc:
            self.store.enqueue("alice", session["id"], "stale", "stale", "answers", {"state_version": 1, "question_id": "q1", "answer": "a"})
        self.assertEqual(exc.exception.code, "stale_state")
        with self.assertRaises(ApiError) as exc:
            self.submit(session, "answers", {"question_id": "old", "answer": "a"})
        self.assertEqual(exc.exception.code, "stale_question")

    def test_user_cap_global_queue_cap_and_single_running_claim(self):
        first = self.create()
        self.submit(first, "perceive")
        with self.assertRaises(ApiError) as exc:
            self.submit(self.create(), "perceive")
        self.assertEqual(exc.exception.code, "user_task_active")
        for index in range(9):
            owner = f"user{index}"
            self.submit(self.create(owner), "perceive", owner=owner)
        with self.assertRaises(ApiError) as exc:
            self.submit(self.create("extra"), "perceive", owner="extra")
        self.assertEqual(exc.exception.status, 429)
        self.assertIsNotNone(self.store.claim_task())
        self.assertIsNone(self.store.claim_task())

    def test_restart_interrupts_running_and_resumes_only_queued(self):
        first = self.create()
        running = self.submit(first, "perceive")
        self.store.claim_task()
        second = self.create("bob")
        queued = self.submit(second, "perceive", owner="bob")
        self.store.interrupt_running()
        self.assertEqual(self.store.get_task("alice", running["id"])["status"], "interrupted")
        self.runner.process_once()
        self.assertEqual(self.store.get_task("bob", queued["id"])["status"], "succeeded")
        self.assertEqual(self.engine.calls, ["perceive"])

    def test_extraction_deletes_raw_material_immediately(self):
        session = self.create()
        self.store.add_upload("alice", session["id"], "f", "f", "a.txt", b"anonymous", "student")
        self.submit(session, "perceive")
        self.runner.process_once()
        self.assertEqual(list(self.store.upload_dir.iterdir()), [])
        self.assertIsNotNone(self.store.get_session("alice", session["id"])["uploads"][0]["deleted_at"])

    def test_failed_guidance_keeps_extracted_state_and_deletes_raw(self):
        session = self.create()
        self.store.add_upload("alice", session["id"], "f", "f", "a.txt", b"anonymous", "student")
        task = self.submit(session, "perceive")
        class PartialFailure(RuntimeError):
            partial_state = {"extracted": True, "status": "ready", "question": None, "versions": []}
            code = "guidance_failed"
            public_message = "引导失败，材料已保留。"
        class FailingEngine:
            def process(self, *args):
                raise PartialFailure("private path and text")
        runner = TaskRunner(self.store, FailingEngine)
        runner.process_once()
        self.assertEqual(self.store.get_task("alice", task["id"])["status"], "failed")
        restored = self.store.get_session("alice", session["id"])
        self.assertEqual(restored["status"], "ready")
        self.assertEqual(restored["state_version"], 3)
        self.assertEqual(list(self.store.upload_dir.iterdir()), [])

    def test_delete_revokes_replay_and_discards_running_result(self):
        session = self.create(key="create-key")
        task = self.submit(session, "perceive")
        claimed = self.store.claim_task()
        args = ("alice", session["id"], "delete", "delete")
        self.assertEqual(self.store.delete_session(*args), self.store.delete_session(*args))
        self.store.finish_task(task["id"], result={**claimed["state"], "plan": "should be discarded"})
        with self.assertRaises(ApiError):
            self.store.get_session("alice", session["id"])
        with self.assertRaises(ApiError):
            self.store.get_task("alice", task["id"])
        with self.store.connection() as conn:
            row = conn.execute("SELECT state FROM sessions WHERE id=?", (session["id"],)).fetchone()
            self.assertEqual(row[0], "{}")
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE session_id=?", (session["id"],)).fetchone()[0], 0)
        with self.assertRaises(ApiError) as exc:
            self.create(key="create-key")
        self.assertEqual(exc.exception.status, 404)

    def test_delete_cancels_queued_without_engine_call(self):
        session = self.create()
        self.submit(session, "perceive")
        self.store.delete_session("alice", session["id"], "delete", "delete")
        self.assertFalse(self.runner.process_once())
        self.assertEqual(self.engine.calls, [])

    def test_retention_deletes_material_at_24h_and_records_at_30d(self):
        session = self.create()
        self.store.add_upload("alice", session["id"], "f", "f", "a.txt", b"anonymous", "unknown")
        self.now += 86400
        self.assertEqual(self.store.cleanup()["files_deleted"], 1)
        self.assertEqual(list(self.store.upload_dir.iterdir()), [])
        self.assertIsNotNone(self.store.get_session("alice", session["id"]))
        self.now += 30*86400
        self.assertEqual(self.store.cleanup()["sessions_deleted"], 1)
        with self.assertRaises(ApiError):
            self.store.get_session("alice", session["id"])

    def test_storage_error_has_no_json_fallback(self):
        self.store.db_path.unlink()
        self.store.db_path.mkdir()
        with self.assertRaises(ApiError) as exc:
            self.store.list_sessions("alice")
        self.assertEqual(exc.exception.status, 503)
        self.assertFalse(any(self.root.glob("*.json")))

    def test_queued_perception_fails_if_materials_expire(self):
        session = self.create()
        self.store.add_upload("alice", session["id"], "f", "f", "a.txt", b"anonymous", "student")
        task = self.submit(session, "perceive")
        self.now += 86400
        self.store.cleanup()
        self.assertFalse(self.runner.process_once())
        saved = self.store.get_task("alice", task["id"])
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error"]["code"], "materials_expired")
        self.assertEqual(self.engine.calls, [])

    def test_failed_model_is_not_retried_and_error_has_no_private_text(self):
        session = self.create()
        task = self.submit(session, "perceive")
        class FailedEngine:
            calls = 0
            def process(self, *args):
                self.calls += 1
                raise RuntimeError("/private/path raw secret text")
        engine = FailedEngine()
        runner = TaskRunner(self.store, lambda: engine)
        self.assertTrue(runner.process_once())
        self.assertFalse(runner.process_once())
        self.assertEqual(engine.calls, 1)
        saved = self.store.get_task("alice", task["id"])
        self.assertEqual(saved["status"], "failed")
        self.assertNotIn("private", json.dumps(saved))

    def test_process_lock_rejects_two_process_owners(self):
        first = ProcessLock(self.root/"worker.lock")
        second = ProcessLock(self.root/"worker.lock")
        first.acquire()
        try:
            with self.assertRaises(RuntimeError):
                second.acquire()
        finally:
            first.release()
        second.acquire()
        second.release()

    def test_evaluation_stays_bound_to_real_saved_version(self):
        session = self.planned()
        data = {"plan_id": "plan1", "output_version": 1, "scores": None, "comment": "仅评论，无评分"}
        result = self.store.record("alice", session["id"], "eval", "eval", data, evaluation=True)[0]
        self.assertIsNone(result["scores"])
        self.assertEqual(result, self.store.record("alice", session["id"], "eval", "eval", data, evaluation=True)[0])
        with self.assertRaises(ApiError):
            self.store.record("alice", session["id"], "bad", "bad", {**data, "output_version": 2}, evaluation=True)
        self.assertEqual(len(self.store.get_session("alice", session["id"])["evaluations"]), 1)

    def test_feedback_rejects_old_version_and_keeps_snapshot(self):
        session = self.planned()
        self.submit(session, "feedback", {"plan_id": "plan1", "source_version": 1, "feedback": "过短"})
        self.runner.process_once()
        view = self.store.get_session("alice", session["id"])
        self.assertEqual(view["knowledge_snapshot"], {"version_id": "k1"})
        self.assertEqual(len(view["versions"]), 2)
        with self.assertRaises(ApiError) as exc:
            self.submit(session, "feedback", {"plan_id": "plan1", "source_version": 1, "feedback": "过短"})
        self.assertEqual(exc.exception.code, "stale_version")


if __name__ == "__main__":
    unittest.main()
