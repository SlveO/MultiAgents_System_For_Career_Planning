from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

import project.api.api as api_mod


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.history = {}

    def run_stream(self, user_input, llm_model=None, session_id="default", trace_id=None):
        self.calls.append(
            {
                "user_input": user_input,
                "llm_model": llm_model,
                "session_id": session_id,
            }
        )
        hist = self.history.setdefault(session_id, [])
        hist.append({"user": user_input, "assistant": "ok"})
        yield {"event": "route", "data": {"mode": "text", "session_id": session_id}}
        yield {"event": "small_model_output", "data": {"mode": "text", "text": "intent"}}
        yield {"event": "rag_references", "data": {"count": 1, "items": [{"role": "r"}]}}
        yield {"event": "llm_input_ready", "data": {"prompt_preview": "intent"}}
        yield {"event": "token", "data": {"token": "ok"}}
        yield {"event": "final", "data": {"text": "ok", "mode": "text", "session_id": session_id}}

    def get_session_history(self, session_id):
        return list(self.history.get(session_id, []))

    def clear_session(self, session_id):
        self.history.pop(session_id, None)


class TestMultimodalApiFlow(unittest.TestCase):
    def setUp(self):
        self._old_pipeline = api_mod.multimodal_pipeline
        self.fake = FakePipeline()
        api_mod.multimodal_pipeline = self.fake
        self.client = TestClient(api_mod.app)

    def tearDown(self):
        api_mod.multimodal_pipeline = self._old_pipeline

    def test_stream_endpoint_with_session(self):
        resp = self.client.post(
            "/v1/multimodal/chat/stream",
            json={"session_id": "s1", "user_input": "hello", "llm_model": "deepseek-chat"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("event: route", resp.text)
        self.assertIn("event: rag_references", resp.text)
        self.assertIn("event: final", resp.text)
        self.assertEqual(self.fake.calls[0]["session_id"], "s1")

    def test_session_get_and_clear(self):
        _ = self.client.post("/v1/multimodal/chat/stream", json={"session_id": "s2", "user_input": "hello"})
        got = self.client.get("/v1/multimodal/chat/session/s2")
        self.assertEqual(got.status_code, 200)
        body = got.json()
        self.assertEqual(body["session_id"], "s2")
        self.assertEqual(len(body["history"]), 1)

        cleared = self.client.delete("/v1/multimodal/chat/session/s2")
        self.assertEqual(cleared.status_code, 200)
        got2 = self.client.get("/v1/multimodal/chat/session/s2")
        self.assertEqual(got2.status_code, 200)
        self.assertEqual(got2.json()["history"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
