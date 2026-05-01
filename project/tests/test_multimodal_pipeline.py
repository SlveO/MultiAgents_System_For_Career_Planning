from __future__ import annotations

import unittest

from project.core.multimodal_pipeline import MultimodalChatPipeline


class FakePerceptionResult:
    def __init__(self, raw_output=""):
        self.raw_output = raw_output
        self.summary = raw_output
        self.facts = []
        self.modality = "image"
        self.confidence = 0.8
        self.missing_info = []
        self.evidence = []


class FakeImageAgent:
    def perceive(self, path, user_goal=None, user_text=None):
        return FakePerceptionResult(f"IMAGE_DESC::{path}::{user_text or ''}")

    def unload(self):
        return None


class FakeImageProcessor:
    def analyze(self, image_path, question=None, context=None):
        return f"IMAGE_DESC::{image_path}::{question or ''}"

    def unload(self):
        return None


class FakeBrainClient:
    def __init__(self):
        self.last_prompt = ""

    def plan_stream(self, prompt, model=None):
        self.last_prompt = prompt
        for token in ["A", "B", "C"]:
            yield token


class FakeRouter:
    def __init__(self, routed):
        self._routed = routed
        self.classifier = self

    def route(self, _input):
        return self._routed

    def validate(self, _metadata):
        return True, "ok"


class TestMultimodalPipeline(unittest.TestCase):
    def test_text_flow_passes_small_model_output_to_llm(self):
        brain = FakeBrainClient()
        routed = {
            "route": "text",
            "target_agent": "agents.perception.TextPerceptionAgent",
            "payload": {"text": "hello"},
            "metadata": {"text_content": "hello"},
        }
        pipeline = MultimodalChatPipeline(
            router=FakeRouter(routed),
            brain_client=brain,
            image_agent=FakeImageAgent(),
            knowledge_base=type("KB", (), {"retrieve": lambda self, query, top_k=4: [{"role": "r", "match_score": "1"}]})(),
        )

        events = list(pipeline.run_stream("hello"))
        event_names = [e["event"] for e in events]

        self.assertIn("route", event_names)
        self.assertIn("small_model_output", event_names)
        self.assertIn("rag_references", event_names)
        self.assertIn("llm_input_ready", event_names)
        self.assertIn("final", event_names)
        self.assertIn("User question:\nhello", brain.last_prompt)
        self.assertIn("Retrieved knowledge references:", brain.last_prompt)
        self.assertEqual(events[-1]["data"]["text"], "ABC")
        self.assertEqual(events[-1]["data"]["session_id"], "default")
        self.assertTrue(events[-1]["data"]["trace_id"])

    def test_multimodal_flow_uses_image_understanding(self):
        brain = FakeBrainClient()
        routed = {
            "route": "multimodal",
            "target_agent": "agents.perception.ImagePerceptionAgent",
            "payload": {"text": "analyze", "images": ["a.jpg"]},
            "metadata": {"text_content": "analyze"},
        }
        pipeline = MultimodalChatPipeline(
            router=FakeRouter(routed),
            brain_client=brain,
            image_agent=FakeImageAgent(),
            knowledge_base=type("KB", (), {"retrieve": lambda self, query, top_k=4: []})(),
        )

        _ = list(pipeline.run_stream('analyze "a.jpg"'))
        self.assertIn("IMAGE_DESC::a.jpg::analyze", brain.last_prompt)

    def test_session_history_is_appended_and_used(self):
        brain = FakeBrainClient()
        routed = {
            "route": "text",
            "target_agent": "agents.perception.TextPerceptionAgent",
            "payload": {"text": "hello again"},
            "metadata": {"text_content": "hello again"},
        }
        pipeline = MultimodalChatPipeline(
            router=FakeRouter(routed),
            brain_client=brain,
            image_agent=FakeImageAgent(),
            knowledge_base=type("KB", (), {"retrieve": lambda self, query, top_k=4: []})(),
        )

        _ = list(pipeline.run_stream("first turn", session_id="s1"))
        _ = list(pipeline.run_stream("second turn", session_id="s1"))
        history = pipeline.get_session_history("s1")

        self.assertEqual(len(history), 2)
        self.assertIn("Recent conversation history:", brain.last_prompt)
        self.assertIn("first turn", brain.last_prompt)

    def test_file_mode_is_processed_and_sent_to_llm(self):
        brain = FakeBrainClient()
        routed = {
            "route": "file",
            "target_agent": "agents.perception.DocumentPerceptionAgent",
            "payload": {"documents": ["a.pdf"], "text": "summarize document"},
            "metadata": {"text_content": "summarize document"},
        }
        pipeline = MultimodalChatPipeline(
            router=FakeRouter(routed),
            brain_client=brain,
            image_agent=FakeImageAgent(),
            knowledge_base=type("KB", (), {"retrieve": lambda self, query, top_k=4: []})(),
        )
        pipeline._understand_documents = lambda paths, user_text="": "DOC_EXTRACT::facts"
        events = list(pipeline.run_stream("a.pdf"))
        self.assertIn("DOC_EXTRACT::facts", brain.last_prompt)
        self.assertEqual(events[-1]["event"], "final")

    def test_audio_video_mode_is_processed_and_sent_to_llm(self):
        brain = FakeBrainClient()
        routed = {
            "route": "audio_video",
            "target_agent": "agents.perception.AudioPerceptionAgent",
            "payload": {"media": ["a.mp3"], "text": "analyze audio"},
            "metadata": {"text_content": "analyze audio"},
        }
        pipeline = MultimodalChatPipeline(
            router=FakeRouter(routed),
            brain_client=brain,
            image_agent=FakeImageAgent(),
            knowledge_base=type("KB", (), {"retrieve": lambda self, query, top_k=4: []})(),
        )
        pipeline._understand_audio_video = lambda paths, user_text="": "AUDIO_EXTRACT::facts"
        events = list(pipeline.run_stream("a.mp3"))
        self.assertIn("AUDIO_EXTRACT::facts", brain.last_prompt)
        self.assertEqual(events[-1]["event"], "final")


if __name__ == "__main__":
    unittest.main(verbosity=2)
