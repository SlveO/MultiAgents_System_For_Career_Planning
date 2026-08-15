from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.experiments.run_completion_experiments import (
    FakeDeepSeekClient,
    run_completion_experiments,
)


class TestCompletionExperiments(unittest.TestCase):
    def test_four_experiment_groups_emit_repeatable_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = run_completion_experiments(output_dir=output_dir)

            json_rows = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
            with (output_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 8)
        self.assertEqual(json_rows, rows)
        self.assertEqual(len(csv_rows), 8)
        self.assertEqual(
            {row["experiment"] for row in rows},
            {
                "template_vs_deepseek",
                "without_vs_with_knowledge",
                "text_vs_text_document",
                "without_vs_with_follow_up",
            },
        )
        for row in rows:
            self.assertEqual(
                set(row["metrics"]),
                {"completeness", "personalization", "actionability", "latency_ms", "user_feedback"},
            )
            self.assertIn(row["metrics"]["user_feedback"], {"过短", "合适", "过于详细"})

    def test_live_mode_uses_the_real_client_boundary_explicitly(self) -> None:
        live_client = FakeDeepSeekClient()
        live_client.api_key = "test-key"
        with tempfile.TemporaryDirectory() as tmp, patch(
            "project.experiments.run_completion_experiments.DeepSeekBrainClient",
            return_value=live_client,
        ) as client_factory:
            rows = run_completion_experiments(output_dir=Path(tmp), live=True)

        client_factory.assert_called_once_with()
        self.assertTrue(all(row["configuration"]["brain_backend"] == "live" for row in rows))

    def test_live_mode_rejects_a_missing_api_key(self) -> None:
        live_client = FakeDeepSeekClient()
        live_client.api_key = ""
        with tempfile.TemporaryDirectory() as tmp, patch(
            "project.experiments.run_completion_experiments.DeepSeekBrainClient",
            return_value=live_client,
        ):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                run_completion_experiments(output_dir=Path(tmp), live=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
