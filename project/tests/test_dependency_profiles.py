from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestDependencyProfiles(unittest.TestCase):
    def test_core_has_document_dependencies_without_gpu_or_api(self):
        content = (ROOT / "requirements.txt").read_text()
        for name in ("httpx", "pydantic-settings", "pypdf", "python-docx"):
            self.assertIn(name, content)
        for name in ("fastapi", "torch"):
            self.assertNotIn(name, content)

    def test_gpu_profile_includes_core_not_api(self):
        content = (ROOT / "requirements-gpu.txt").read_text()
        self.assertIn("-r requirements.txt", content)
        self.assertIn("transformers", content)
        self.assertNotIn("requirements-api.txt", content)
        self.assertNotIn("lm-format-enforcer", content)

    def test_retired_profiles_and_product_stacks_are_removed(self):
        for path in ("requirements-api.txt", "requirements-mvp.txt",
                     "requirements-docker.txt", "Dockerfile", "docker-compose.yml",
                     "project/api/api.py", "project/experiments/run_architecture_experiments.py"):
            self.assertFalse((ROOT / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
