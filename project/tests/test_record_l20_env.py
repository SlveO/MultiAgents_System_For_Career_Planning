"""Sanitization and guard tests for the L20 environment record script."""

import importlib.util
import socket
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "models" / "record_l20_env.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("record_l20_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRedaction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _load_script()

    def test_redacts_hostname_username_and_home_path(self):
        host = socket.gethostname()
        text = (
            f"host={host} user=alice /home/alice/repo "
            f"/home/bob/other sk-abcdef0123456789abcdef"
        )
        out = self.script.redact(text)
        self.assertNotIn(host, out)
        self.assertIn("[REDACTED_HOST]", out)
        self.assertNotIn("/home/alice", out)
        self.assertNotIn("/home/bob", out)
        self.assertNotIn("sk-abcdef0123456789abcdef", out)
        self.assertIn("[REDACTED_KEY]", out)

    def test_markdown_table_has_expected_rows(self):
        record = {
            "os": "Ubuntu 24.04",
            "kernel": "Linux x",
            "python": "Python 3.11",
            "nvidia_smi": "name",
            "torch": "torch 2.x",
            "disk": "ok",
            "network": "ok",
        }
        md = self.script.to_markdown(record)
        for key in ("Ubuntu 版本", "内核", "驱动 / 显存", "PyTorch / CUDA", "网络"):
            self.assertIn(key, md)

    def test_collect_refuses_non_linux(self):
        # collect() guards before running any command on this machine.
        # Mock the platform so the test is environment-agnostic.
        import unittest.mock as mock

        with mock.patch.object(
            self.script.platform, "system", return_value="Windows"
        ):
            with self.assertRaises(SystemExit):
                self.script.collect()


if __name__ == "__main__":
    unittest.main()
