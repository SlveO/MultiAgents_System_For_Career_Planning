from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.core.schemas import TaskRequest
from project.core.run_logging import JsonlRunLogger
from project.orchestrator import CareerOrchestrator


class FakeDeepSeekClient:
    model_name = "deepseek-v4-flash"

    def plan(self, prompt: str, model: str | None = None) -> str:
        self.prompt = prompt
        return json.dumps(
            {
                "user_facing_advice": "先完成岗位能力盘点，再制作可展示项目。",
                "target_roles": ["数据分析师"],
                "gap_analysis": ["缺少可量化项目"],
                "roadmap_30_90_180": [
                    {"period": "30d", "objective": "完成盘点", "deliverables": ["JD清单"], "metrics": ["10份JD"]},
                    {"period": "90d", "objective": "完成项目", "deliverables": ["作品集"], "metrics": ["2个项目"]},
                    {"period": "180d", "objective": "完成投递", "deliverables": ["复盘表"], "metrics": ["30次投递"]},
                ],
                "learning_resources": ["SQLBolt"],
                "next_actions": ["今天拆解两份JD"],
                "risk_flags": ["目标过多"],
                "follow_up_questions": [],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        )

    def plan_stream(self, prompt: str, model: str | None = None):
        yield self.plan(prompt, model)


class TestCompletionFlow(unittest.TestCase):
    def test_text_and_document_flow_works_without_initializing_image_stack(self) -> None:
        answers = {
            "education": "本科大三",
            "major": "统计学",
            "skills": "Python、SQL",
            "interests": "数据分析",
            "target_role": "数据分析师",
            "time_budget": "每周10小时",
            "preference": "上海互联网",
            "constraints": "项目经验不足",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "resume.txt"
            document.write_text("熟悉 Python 和 SQL，想做数据分析。", encoding="utf-8")
            brain = FakeDeepSeekClient()

            with patch(
                "project.agents.perception.image_agent._load_image_processor",
                side_effect=ImportError("torch unavailable"),
            ):
                orchestrator = CareerOrchestrator(
                    db_path=str(root / "sessions.db"),
                    brain_client=brain,
                    run_logger=JsonlRunLogger(root / "runs.jsonl"),
                )
                request = TaskRequest(
                        session_id="demo",
                        user_goal="获得数据分析实习",
                        text_input="邮箱 demo@example.com",
                        document_paths=[str(document)],
                        follow_up_answers=answers,
                    )
                response = orchestrator.run(request)
                orchestrator.submit_feedback("demo", "合适")

            history = orchestrator.memory.get_session_history("demo")
            log_record = json.loads((root / "runs.jsonl").read_text(encoding="utf-8").strip())

        self.assertEqual(response.served_by, "cloud_brain")
        self.assertEqual(response.profile.target_role, "数据分析师")
        self.assertEqual(response.profile.skills, ["Python", "SQL"])
        self.assertEqual([item.period for item in response.roadmap_30_90_180], ["30d", "90d", "180d"])
        self.assertTrue(response.knowledge_hit_ids)
        self.assertEqual(log_record["knowledge_result_ids"], response.knowledge_hit_ids)
        self.assertIn("本科大三", brain.prompt)
        self.assertIsNone(orchestrator.image_agent)
        persisted = json.dumps(history, ensure_ascii=False)
        self.assertNotIn(str(document), persisted)
        self.assertNotIn("demo@example.com", persisted)
        self.assertEqual(log_record["feedback"], "合适")


if __name__ == "__main__":
    unittest.main(verbosity=2)
