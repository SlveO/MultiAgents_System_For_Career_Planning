from __future__ import annotations

import gc
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, List
from unittest.mock import patch

import httpx

from project.core.brain_client import BrainResponseError, DeepSeekBrainClient
from project.core.run_logging import JsonlRunLogger
from project.core.schemas import TaskRequest
from project.orchestrator import CareerOrchestrator


def _plan_json(advice: str, gap_count: int = 2, deliverable_count: int = 2) -> str:
    return json.dumps(
        {
            "user_facing_advice": advice,
            "target_roles": ["数据分析师"],
            "gap_analysis": [f"差距{i}" for i in range(gap_count)],
            "roadmap_30_90_180": [
                {
                    "period": period,
                    "objective": f"{period} 目标",
                    "deliverables": [f"交付{i}" for i in range(deliverable_count)],
                    "metrics": ["指标"],
                }
                for period in ("30d", "90d", "180d")
            ],
            "learning_resources": ["SQLBolt"],
            "next_actions": ["拆解 JD"],
            "risk_flags": ["项目不足"],
            "follow_up_questions": [],
            "confidence": 0.8,
        },
        ensure_ascii=False,
    )


class ScriptedBrainClient:
    """Deterministic fake: uniform output for the first call, tier-specific
    outputs for feedback calls; optionally fails feedback calls."""

    def __init__(self, fail_feedback: bool = False) -> None:
        self.calls: List[str] = []
        self.fail_feedback = fail_feedback

    def plan(self, prompt: str, model: str | None = None) -> str:
        self.calls.append(prompt)
        if "反馈档位: 过短" in prompt:
            if self.fail_feedback:
                raise BrainResponseError("fake adjust failure")
            return _plan_json("扩充后的建议", gap_count=4, deliverable_count=3)
        if "反馈档位: 过于详细" in prompt:
            if self.fail_feedback:
                raise BrainResponseError("fake adjust failure")
            return _plan_json("精简后的建议", gap_count=1, deliverable_count=1)
        return _plan_json("统一建议")

    def plan_stream(self, prompt: str, model: str | None = None) -> Iterable[str]:
        yield self.plan(prompt, model)


class AdjustTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.brain = ScriptedBrainClient()
        self.logger = JsonlRunLogger(base / "runs.jsonl")
        self.orchestrator = CareerOrchestrator(
            db_path=str(base / "memory.db"),
            brain_client=self.brain,
            run_logger=self.logger,
        )
        self.req = TaskRequest(
            session_id="t1",
            user_goal="获得数据分析实习",
            text_input="会 Python 和 SQL",
            planner_mode="deepseek",
            use_knowledge=True,
        )

    def tearDown(self) -> None:
        # SQLite connections are closed by GC; Windows keeps locked files
        # otherwise, so force collection and tolerate leftovers.
        gc.collect()
        shutil.rmtree(self._tmp.name, ignore_errors=True)

    def _logs(self) -> List[dict]:
        path = self.logger.path
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_suitable_feedback_records_without_regeneration(self) -> None:
        original = self.orchestrator.run(self.req)
        adjusted, error = self.orchestrator.adjust_plan("t1", "合适")
        self.assertIs(adjusted, original)
        self.assertIsNone(error)
        self.assertEqual(len(self.brain.calls), 1)
        logs = self._logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["feedback"], "合适")
        self.assertNotIn("feedback_adjusted", logs[0])

    def test_short_feedback_regenerates_expanded_plan(self) -> None:
        self.orchestrator.run(self.req)
        adjusted, error = self.orchestrator.adjust_plan("t1", "过短")
        self.assertIsNone(error)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.user_facing_advice, "扩充后的建议")
        self.assertEqual(len(self.brain.calls), 2)
        self.assertIn("反馈档位: 过短", self.brain.calls[1])
        logs = self._logs()
        self.assertEqual(logs[0]["feedback"], "过短")
        self.assertTrue(logs[0]["feedback_adjusted"])
        self.assertNotIn("error", logs[0])
        # Pending run consumed: a second adjust has nothing to work on.
        second, second_error = self.orchestrator.adjust_plan("t1", "过短")
        self.assertIsNone(second)
        self.assertIsNone(second_error)

    def test_detailed_feedback_regenerates_condensed_plan(self) -> None:
        self.orchestrator.run(self.req)
        adjusted, error = self.orchestrator.adjust_plan("t1", "过于详细")
        self.assertIsNone(error)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.user_facing_advice, "精简后的建议")
        self.assertIn("反馈档位: 过于详细", self.brain.calls[1])

    def test_adjust_failure_keeps_original_and_records_error(self) -> None:
        failing = ScriptedBrainClient(fail_feedback=True)
        self.orchestrator.cloud_brain = failing
        original = self.orchestrator.run(self.req)
        adjusted, error = self.orchestrator.adjust_plan("t1", "过短")
        # 手册 3.2: 保留原结果并记录失败。
        self.assertIs(adjusted, original)
        self.assertIsNotNone(error)
        self.assertIn("BRAIN_RESPONSE", error)
        logs = self._logs()
        self.assertEqual(logs[0]["feedback"], "过短")
        self.assertIn("error", logs[0])
        self.assertNotIn("feedback_adjusted", logs[0])

    def test_no_pending_run_records_and_returns_none(self) -> None:
        adjusted, error = self.orchestrator.adjust_plan("unknown", "过短")
        self.assertIsNone(adjusted)
        self.assertIsNone(error)

    def test_invalid_feedback_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.orchestrator.adjust_plan("t1", "太长")


class FakeHttpClient:
    def __init__(self, *, response=None) -> None:
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, _url, *, headers, json):
        return self.response


class TestDeepSeekUsage(unittest.TestCase):
    def _build_client(self) -> DeepSeekBrainClient:
        client = DeepSeekBrainClient.__new__(DeepSeekBrainClient)
        client.api_key = "test-key"
        client.base_url = "https://api.deepseek.com"
        client.default_model = "deepseek-v4-flash"
        client.timeout = 1.0
        client.last_usage = {}
        return client

    def test_plan_records_token_usage(self) -> None:
        response = httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "plan"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        client = self._build_client()
        with patch("httpx.Client", return_value=FakeHttpClient(response=response)):
            client.plan("hi")
        self.assertEqual(client.last_usage.get("total_tokens"), 15)


if __name__ == "__main__":
    unittest.main()
