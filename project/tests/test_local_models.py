"""Registry and adapter tests for the frozen local research models.

These tests must pass in the default (CPU, no torch) environment: they verify
that the module imports cleanly, the registry pins exact revisions, and the
missing-weights path fails before any GPU dependency is imported.
"""

import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from project.experiments import local_models
from project.experiments.local_models import (
    FROZEN_MODELS,
    ModelNotDownloadedError,
    get_entry,
    local_model_dir,
)


class TestFrozenRegistry(unittest.TestCase):
    def test_three_frozen_models_with_expected_roles(self):
        self.assertEqual(len(FROZEN_MODELS), 3)
        roles = {m["role"] for m in FROZEN_MODELS}
        self.assertEqual(roles, {"monolithic", "perceiver", "reasoner"})

    def test_each_entry_has_pinned_revision_and_kind(self):
        for entry in FROZEN_MODELS:
            revision = entry["revision"]
            self.assertEqual(len(revision), 40, entry["model_id"])
            self.assertTrue(
                all(c in "0123456789abcdef" for c in revision), entry["model_id"]
            )
            self.assertIn(entry["kind"], {"vl", "text"}, entry["model_id"])
            self.assertTrue(entry["pinned_at"], entry["model_id"])

    def test_group_coverage_matches_experiment_design(self):
        # Monolithic uses the 8B VL model; modular groups share 2B/4B.
        monolithic = get_entry("Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(monolithic["role"], "monolithic")
        self.assertEqual(monolithic["kind"], "vl")
        perceiver = get_entry("Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(perceiver["kind"], "vl")
        reasoner = get_entry("Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(reasoner["kind"], "text")

    def test_local_model_dir_is_under_models_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = local_model_dir(
                "Qwen/Qwen3-4B-Instruct-2507", models_dir=Path(tmp)
            )
            self.assertEqual(target, Path(tmp) / "Qwen3-4B-Instruct-2507")

    def test_get_entry_rejects_unknown_model(self):
        with self.assertRaises(KeyError):
            get_entry("Qwen/does-not-exist")


class TestAdapterWithoutGPU(unittest.TestCase):
    def test_module_imports_without_torch(self):
        # The lazy-import contract must hold in any environment, including
        # hosts where the GPU stack is installed. Block torch/transformers
        # imports in a clean subprocess and import the module there.
        import subprocess
        import sys

        code = (
            "import sys; "
            "sys.modules['torch'] = None; "
            "sys.modules['transformers'] = None; "
            "from project.experiments import local_models; "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_weights_fails_before_gpu_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = local_models.LocalModelAdapter(
                "Qwen/Qwen3-4B-Instruct-2507",
                models_dir=Path(tmp),
            )
            with self.assertRaises(ModelNotDownloadedError):
                adapter.load()

    def test_generate_before_load_raises(self):
        adapter = local_models.LocalModelAdapter(
            "Qwen/Qwen3-4B-Instruct-2507"
        )
        with self.assertRaises(RuntimeError):
            adapter.generate("prompt")

    def test_deterministic_defaults(self):
        adapter = local_models.LocalModelAdapter(
            "Qwen/Qwen3-4B-Instruct-2507"
        )
        self.assertEqual(adapter.seed, 42)
        self.assertEqual(adapter.max_new_tokens, 256)


if __name__ == "__main__":
    unittest.main()
