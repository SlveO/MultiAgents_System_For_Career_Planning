from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from project.core.feedback import FEEDBACK_OPTIONS, collect_feedback
from project.core.privacy import redact_data, redact_text
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import CareerPlanResponse, FeedbackRequest, TaskRequest, UserProfile
from project.core.session_memory import SessionMemory


class TestPrivacyAndLogging(unittest.TestCase):
    def test_redacts_personal_identifiers_and_windows_username(self) -> None:
        raw = (
            "姓名：张三，手机号13812345678，邮箱 student@example.com，"
            "学号：2026123456，身份证号11010520000101123X，"
            r"文件 C:\Users\Alice\Documents\resume.docx"
        )

        redacted = redact_text(raw)

        for secret in ["张三", "13812345678", "student@example.com", "2026123456", "11010520000101123X", "Alice"]:
            self.assertNotIn(secret, redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)
        self.assertIn(r"C:\Users\[REDACTED_USER]\Documents", redacted)

    def test_nested_redaction_drops_raw_content_and_paths(self) -> None:
        payload = {
            "text_input": "联系我 student@example.com",
            "document_paths": [r"C:\Users\Alice\resume.docx"],
            "perception_results": [{"raw_output": "完整简历内容", "summary": "姓名：张三"}],
        }

        safe = redact_data(payload)

        self.assertNotIn("document_paths", safe)
        self.assertNotIn("raw_output", safe["perception_results"][0])
        self.assertNotIn("student@example.com", json.dumps(safe, ensure_ascii=False))
        self.assertNotIn("张三", json.dumps(safe, ensure_ascii=False))

    def test_feedback_is_limited_to_three_levels(self) -> None:
        self.assertEqual(FEEDBACK_OPTIONS, ("过短", "合适", "过于详细"))
        self.assertEqual(FeedbackRequest(session_id="s1", feedback="合适").feedback, "合适")
        with self.assertRaises(ValidationError):
            FeedbackRequest(session_id="s1", feedback="写得不错")

        values = iter(["无效", "2"])
        self.assertEqual(collect_feedback(lambda _prompt: next(values)), "合适")

    def test_jsonl_run_contains_required_redacted_fields(self) -> None:
        request = TaskRequest(
            session_id="session-1",
            user_goal="姓名：张三想做数据分析",
            text_input="邮箱 student@example.com",
            document_paths=[r"C:\Users\Alice\resume.docx"],
            follow_up_answers={"skills": "Python、SQL"},
        )
        response = CareerPlanResponse(
            session_id="session-1",
            intent="planning",
            profile=UserProfile(skills=["Python", "SQL"], target_role="数据分析师"),
            target_roles=["数据分析师"],
            user_facing_advice="下一步联系 student@example.com",
            knowledge_hit_ids=["career-002"],
            served_by="cloud_brain",
            latency_ms=321,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            logger = JsonlRunLogger(path)
            logger.log_run(request, response, feedback="合适", model_name="deepseek-v4-flash")
            record = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertEqual(record["session_id"], "session-1")
        self.assertEqual(record["knowledge_result_ids"], ["career-002"])
        self.assertEqual(record["model"], "deepseek-v4-flash")
        self.assertEqual(record["feedback"], "合适")
        self.assertEqual(record["latency_ms"], 321)
        self.assertIn("timestamp", record)
        serialized = json.dumps(record, ensure_ascii=False)
        for secret in ["张三", "student@example.com", "Alice", "完整简历内容"]:
            self.assertNotIn(secret, serialized)

    def test_session_history_is_chronological_and_can_be_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = SessionMemory(str(Path(tmp) / "sessions.db"))
            memory.append_interaction("s1", {"turn": 1}, {"answer": "first"})
            memory.append_interaction("s1", {"turn": 2}, {"answer": "second"})

            history = memory.get_session_history("s1")
            self.assertEqual([item["request"]["turn"] for item in history], [1, 2])

            memory.clear_session_history("s1")
            self.assertEqual(memory.get_session_history("s1"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
