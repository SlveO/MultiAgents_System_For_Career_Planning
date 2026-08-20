"""Platform guard and registry tests for the L20 download script.

The script lives outside the package, so it is loaded from its path. No
network or GPU dependency is touched: the guard must refuse Windows before
anything else runs.
"""

import importlib.util
import unittest
from pathlib import Path

from project.experiments.local_models import FROZEN_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "models" / "download_research_models.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "download_research_models", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPlatformGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()

    def test_windows_is_refused(self):
        with self.assertRaises(self.script.PlatformGuardError):
            self.script.ensure_allowed_platform("Windows")

    def test_linux_is_allowed(self):
        self.script.ensure_allowed_platform("Linux")  # must not raise

    def test_guard_runs_before_download(self):
        # main() raises the guard error directly; the CLI wrapper maps it to
        # exit code 2 so a Windows run stops before touching modelscope.
        # Patch the platform so this test never downloads on any host.
        import unittest.mock as mock

        with mock.patch.object(
            self.script.platform, "system", return_value="Windows"
        ):
            with self.assertRaises(self.script.PlatformGuardError):
                self.script.main(["--models-dir", "models"])

    def test_script_registry_matches_module_registry(self):
        # Single source of truth: the script imports the module registry.
        self.assertEqual(self.script.FROZEN_MODELS, FROZEN_MODELS)


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()

    def test_manifest_records_revision_and_size(self):
        import json
        import tempfile

        # Manifest content is built from the registry; simulate the write
        # with one fake entry directory. modelscope stays a lazy dependency
        # and is never imported by this path.
        models = FROZEN_MODELS

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for entry in models:
                target = root / entry["model_id"].split("/")[1]
                target.mkdir(parents=True)
                (target / "config.json").write_text("{}", encoding="utf-8")
            manifest = self.script.write_manifest(models, root)
            records = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertEqual(len(record["revision"]), 40)
                self.assertTrue(record["size_bytes"] > 0)
                self.assertIn("models/", record["local_dir"])


if __name__ == "__main__":
    unittest.main()
