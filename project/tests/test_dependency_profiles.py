from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _requirements(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class TestDependencyProfiles(unittest.TestCase):
    def test_core_profile_is_small_and_has_document_dependencies(self):
        content = _requirements("requirements.txt")
        self.assertIn("httpx", content)
        self.assertIn("pydantic-settings", content)
        self.assertIn("pypdf", content)
        self.assertIn("python-docx", content)
        self.assertNotIn("fastapi", content)
        self.assertNotIn("torch", content)

    def test_api_profile_includes_core_profile(self):
        content = _requirements("requirements-api.txt")
        self.assertIn("-r requirements.txt", content)
        self.assertIn("fastapi", content)
        self.assertIn("uvicorn", content)

    def test_gpu_profile_includes_api_profile(self):
        content = _requirements("requirements-gpu.txt")
        self.assertIn("-r requirements-api.txt", content)
        self.assertIn("transformers", content)
        self.assertIn("sentence-transformers", content)

    def test_legacy_profiles_and_windows_wheel_are_removed(self):
        self.assertFalse((ROOT / "requirements-mvp.txt").exists())
        self.assertFalse((ROOT / "requirements-docker.txt").exists())
        self.assertNotIn("bitsandbytes @", _requirements("requirements-gpu.txt"))

    def test_docker_uses_api_profile(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY requirements-api.txt", dockerfile)
        self.assertIn("-r requirements-api.txt", dockerfile)


if __name__ == "__main__":
    unittest.main()
