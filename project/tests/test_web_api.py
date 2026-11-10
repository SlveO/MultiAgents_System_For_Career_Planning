from __future__ import annotations

from pathlib import Path
import asyncio
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from project.web.app import create_app
except ImportError:
    TestClient = None

from project.tests.test_web_storage import FakeEngine


@unittest.skipIf(TestClient is None, "Install requirements-web.txt for Web API tests")
class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.engine = FakeEngine()
        self.app = create_app(db_path=root/"web.db", upload_dir=root/"uploads", service_token="t"*48, engine_factory=lambda: self.engine, start_worker=False)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.counter = 0

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def headers(self, owner="alice", key=None):
        self.counter += 1
        return {"Authorization": "Bearer "+"t"*48, "X-Career-User": owner, "Idempotency-Key": key or str(self.counter)}

    def create(self, owner="alice"):
        response = self.client.post("/v1/sessions", headers=self.headers(owner), json={"user_goal": "匿名职业规划", "text_input": "本科学生"})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def get(self, session, owner="alice"):
        return self.client.get(f'/v1/sessions/{session["id"]}', headers=self.headers(owner))

    def job(self, session, route, fields=None):
        current = self.get(session).json()
        response = self.client.post(f'/v1/sessions/{session["id"]}/{route}', headers=self.headers(), json={"state_version": current["state_version"], **(fields or {})})
        self.assertEqual(response.status_code, 202, response.text)
        self.app.state.runner.process_once()
        return response.json()

    def planned(self):
        session = self.create()
        self.job(session, "perceive")
        self.job(session, "answers", {"question_id": "q1", "answer": "Python"})
        self.job(session, "plans")
        return session

    def test_identity_requires_service_token_and_user(self):
        for headers in ({}, {"X-Career-User": "alice"}, {"Authorization": "Bearer wrong", "X-Career-User": "alice"}, {"Authorization": "Bearer "+"t"*48}):
            response = self.client.get("/v1/sessions", headers=headers)
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get("/healthz").status_code, 200)

    def test_all_owner_scoped_reads_exports_mutations(self):
        session = self.planned()
        prefix = f'/v1/sessions/{session["id"]}'
        for path in (prefix, prefix+"/versions", prefix+"/export", prefix+"/export?format=markdown"):
            response = self.client.get(path, headers=self.headers("bob"))
            self.assertEqual(response.status_code, 404, path)
        response = self.client.post(prefix+"/evaluations", headers=self.headers("bob"), json={"plan_id": "plan1", "output_version": 1})
        self.assertEqual(response.status_code, 404)
        response = self.client.post(prefix+"/uploads", headers=self.headers("bob"), files={"file": ("a.txt", b"anonymous")})
        self.assertEqual(response.status_code, 404)
        response = self.client.delete(prefix, headers=self.headers("bob"))
        self.assertEqual(response.status_code, 404)

    def test_mutations_require_idempotency_and_replay_exact_result(self):
        headers = self.headers(key="stable")
        first = self.client.post("/v1/sessions", headers=headers, json={"user_goal": "匿名规划"})
        again = self.client.post("/v1/sessions", headers=headers, json={"user_goal": "匿名规划"})
        self.assertEqual(first.status_code, 201)
        self.assertEqual(first.json(), again.json())
        mismatch = self.client.post("/v1/sessions", headers=headers, json={"user_goal": "不同目标"})
        self.assertEqual(mismatch.status_code, 409)
        headers.pop("Idempotency-Key")
        self.assertEqual(self.client.post("/v1/sessions", headers=headers, json={"user_goal": "缺键"}).status_code, 400)

    def test_refresh_restores_pending_question_without_model_call(self):
        session = self.create()
        task = self.job(session, "perceive")
        self.assertEqual(self.get(session).json()["question"]["id"], "q1")
        self.assertEqual(self.get(session).json()["question"]["id"], "q1")
        response = self.client.get('/v1/tasks/'+task["id"], headers=self.headers())
        self.assertEqual(response.json()["status"], "succeeded")
        self.assertEqual(self.engine.calls, ["perceive"])
        self.assertEqual(self.client.get('/v1/tasks/'+task["id"], headers=self.headers("bob")).status_code, 404)

    def test_stale_answer_returns_conflict_and_never_reperceives(self):
        session = self.create()
        self.job(session, "perceive")
        response = self.client.post(f'/v1/sessions/{session["id"]}/answers', headers=self.headers(), json={"state_version": 1, "question_id": "q1", "answer": "技能"})
        self.assertEqual(response.status_code, 409)
        self.job(session, "answers", {"question_id": "q1", "answer": "Python"})
        self.assertEqual(self.engine.calls, ["perceive", "answers"])

    def test_upload_metadata_does_not_reveal_path_and_replay_is_durable(self):
        session = self.create()
        path = f'/v1/sessions/{session["id"]}/uploads'
        headers = self.headers(key="file")
        response = self.client.post(path, headers=headers, files={"file": ("../../anonymous.txt", b"anonymous")}, data={"subject": "job"})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["name"], "anonymous.txt")
        self.assertEqual(response.json()["subject"], "job")
        self.assertNotIn("path", response.json())
        again = self.client.post(path, headers=headers, files={"file": ("../../anonymous.txt", b"anonymous")}, data={"subject": "job"})
        self.assertEqual(response.json(), again.json())
        self.job(session, "perceive")
        self.assertEqual(list(self.app.state.store.upload_dir.iterdir()), [])

    def test_evaluations_no_default_scores_and_strict_validation(self):
        session = self.planned()
        path = f'/v1/sessions/{session["id"]}/evaluations'
        response = self.client.post(path, headers=self.headers(), json={"plan_id": "plan1", "output_version": 1, "comment": "仅匿名评论"})
        self.assertEqual(response.status_code, 201)
        self.assertIsNone(response.json()["scores"])
        for value in (True, "5", 5.0, 0, 6):
            response = self.client.post(path, headers=self.headers(), json={"plan_id": "plan1", "output_version": 1, "scores": {"clarity": value}})
            self.assertEqual(response.status_code, 422)
        response = self.client.post(path, headers=self.headers(), json={"plan_id": "other", "output_version": 1})
        self.assertEqual(response.status_code, 409)

    def test_export_uses_saved_report_and_snapshot(self):
        session = self.planned()
        prefix = f'/v1/sessions/{session["id"]}'
        response = self.client.get(prefix+"/export?format=markdown", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertIn("保存的报告", response.text)
        response = self.client.get(prefix+"/export?format=json", headers=self.headers())
        self.assertEqual(response.json()["knowledge_snapshot"]["version_id"], "k1")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_delete_revokes_all_reads_and_idempotently_replays(self):
        session = self.planned()
        prefix = f'/v1/sessions/{session["id"]}'
        headers = self.headers(key="del")
        response = self.client.delete(prefix, headers=headers)
        self.assertEqual(response.json(), {"deleted": True})
        self.assertEqual(self.client.delete(prefix, headers=headers).json(), {"deleted": True})
        for path in (prefix, prefix+"/versions", prefix+"/export"):
            self.assertEqual(self.client.get(path, headers=self.headers()).status_code, 404)

    def test_database_failure_is_explicit_503(self):
        self.app.state.store.db_path.unlink()
        self.app.state.store.db_path.mkdir()
        response = self.client.get("/v1/sessions", headers=self.headers())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "storage_unavailable")

    def test_non_ascii_service_authorization_is_unauthorized(self):
        response = self.client.get("/v1/sessions", headers={"Authorization": b"Bearer \xff", "X-Career-User": "alice"})
        self.assertEqual(response.status_code, 401)

    def test_chunked_upload_limit_closes_partial_temporary_file(self):
        from project.web.body_limit import BodyLimitMiddleware
        from starlette.requests import Request
        from starlette.responses import Response
        opened = []
        factory = tempfile.SpooledTemporaryFile
        def temporary_file(*args, **kwargs):
            handle = factory(*args, **kwargs)
            opened.append(handle)
            return handle
        async def scenario():
            messages = iter([
                {"type": "http.request", "body": b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\nContent-Type: text/plain\r\n\r\nanonymous', "more_body": True},
                {"type": "http.request", "body": b"x"*200, "more_body": False},
            ])
            async def receive():
                return next(messages)
            sent = []
            async def send(message):
                sent.append(message)
            async def parse(scope, receive, send):
                try:
                    await Request(scope, receive).form()
                except Exception:
                    response = Response(status_code=400)
                    await response(scope, receive, send)
            scope = {"type": "http", "method": "POST", "path": "/uploads", "headers": [(b"content-type", b"multipart/form-data; boundary=b")], "app": self.app}
            await BodyLimitMiddleware(parse, max_bytes=180)(scope, receive, send)
            self.assertEqual(sent[0]["status"], 413)
        with patch("starlette.formparsers.SpooledTemporaryFile", side_effect=temporary_file):
            asyncio.run(scenario())
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)


if __name__ == "__main__":
    unittest.main()
