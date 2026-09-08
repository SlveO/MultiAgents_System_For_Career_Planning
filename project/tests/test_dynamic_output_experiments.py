from __future__ import annotations

import gc
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from project.experiments.run_dynamic_output_experiments import (
    _direction_matches,
    _load_cases,
    run_dynamic_output_experiments,
)
from project.orchestrator import CareerOrchestrator


class TestCases(unittest.TestCase):
    def test_case_file_has_twenty_cases_with_valid_feedback(self) -> None:
        from project.experiments.run_dynamic_output_experiments import DEFAULT_CASES

        cases = _load_cases(DEFAULT_CASES)
        self.assertEqual(len(cases), 20)
        short = [c for c in cases if c["feedback"] == "过短"]
        detailed = [c for c in cases if c["feedback"] == "过于详细"]
        self.assertEqual(len(short), 10)
        self.assertEqual(len(detailed), 10)
        for case in cases:
            self.assertTrue(case["user_goal"])
            self.assertTrue(case["follow_up_answers"])

    def test_case_file_rejects_wrong_feedback_values(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            json.dump(
                [{"case_id": f"c{i}", "user_goal": "g", "text_input": "", "follow_up_answers": {}, "feedback": "过短" if i < 10 else "合适"} for i in range(20)],
                handle,
                ensure_ascii=False,
            )
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                _load_cases(path)
        finally:
            path.unlink(missing_ok=True)


class TestDirectionMatch(unittest.TestCase):
    def test_short_feedback_matches_when_expanded(self) -> None:
        from project.experiments.run_dynamic_output_experiments import FakeTieredDeepSeekClient

        fake = FakeTieredDeepSeekClient()
        tmp = tempfile.TemporaryDirectory()
        try:
            orchestrator = CareerOrchestrator(
                db_path=str(Path(tmp.name) / "mem.db"), brain_client=fake
            )
            from project.core.schemas import TaskRequest

            req = TaskRequest(
                session_id="d1",
                user_goal="获得数据分析实习",
                text_input="会 Python",
                planner_mode="deepseek",
                use_knowledge=True,
            )
            uniform = orchestrator.run(req)
            adjusted, error = orchestrator.adjust_plan("d1", "过短")
            self.assertIsNone(error)
            self.assertTrue(_direction_matches("过短", uniform, adjusted))
            # Pending run was consumed by the first adjust: nothing left to adjust.
            detailed_adjusted, error2 = orchestrator.adjust_plan("d1", "过于详细")
            self.assertIsNone(detailed_adjusted)
            self.assertIsNone(error2)
        finally:
            gc.collect()
            shutil.rmtree(tmp.name, ignore_errors=True)


class TestOfflineRunner(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        gc.collect()
        shutil.rmtree(self._tmp.name, ignore_errors=True)

    def test_offline_run_emits_forty_rows_with_artifacts(self) -> None:
        rows = run_dynamic_output_experiments(self.output_dir)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(1 for r in rows if r["group"] == "uniform"), 20)
        self.assertEqual(sum(1 for r in rows if r["group"] == "feedback_adjusted"), 20)
        for row in rows:
            self.assertEqual(row["brain_backend"], "offline-fake")
            self.assertEqual(row["adjust_error"], "")
        adjusted = [r for r in rows if r["group"] == "feedback_adjusted"]
        self.assertTrue(all(r["direction_match"] for r in adjusted))
        for path in (
            "results.json",
            "results.csv",
            "scores_template.csv",
            "summary.json",
            "summary.md",
            "chart.svg",
        ):
            self.assertTrue((self.output_dir / path).exists(), path)

    def test_offline_rows_must_be_marked_fake(self) -> None:
        rows = run_dynamic_output_experiments(self.output_dir)
        self.assertTrue(all(r["brain_backend"] == "offline-fake" for r in rows))


if __name__ == "__main__":
    unittest.main()
