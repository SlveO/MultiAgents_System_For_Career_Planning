from __future__ import annotations

import os
import unittest
from unittest.mock import patch

try:
    from project.web.__main__ import main
    import uvicorn
except ImportError:
    main = None


@unittest.skipIf(main is None, "Web dependencies are optional for CLI tests")
class TestWebStartup(unittest.TestCase):
    def test_rejects_conflicting_gpu_before_starting_service(self):
        with patch.dict(os.environ, {"LOCAL_MODEL_DEVICE": "cuda:0"}), patch("project.web.__main__.load_dotenv"), patch("uvicorn.run") as run:
            with self.assertRaisesRegex(RuntimeError, "GPU1"):
                main()
            run.assert_not_called()

    def test_defaults_to_gpu1_and_disables_retries_on_loopback(self):
        with patch.dict(os.environ, {}, clear=True), patch("project.web.__main__.load_dotenv"), patch("project.web.app.create_app") as create, patch("uvicorn.run") as run:
            main()
            self.assertEqual(os.environ["LOCAL_MODEL_DEVICE"], "cuda:1")
            self.assertEqual(os.environ["BRAIN_RETRY_TIMES"], "0")
            self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
            self.assertEqual(run.call_args.kwargs["workers"], 1)
            create.assert_called_once()

    def test_local_demo_can_choose_a_separate_loopback_port(self):
        with patch.dict(os.environ, {"LOCAL_MODEL_DEVICE": "cuda:1", "CAREER_WEB_PORT": "8788"}), patch("project.web.__main__.load_dotenv"), patch("project.web.app.create_app"), patch("uvicorn.run") as run:
            main()
            self.assertEqual(run.call_args.kwargs["port"], 8788)
            self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")

    def test_rejects_privileged_or_invalid_port(self):
        for port in ("80", "65536"):
            with self.subTest(port=port), patch.dict(os.environ, {"LOCAL_MODEL_DEVICE": "cuda:1", "CAREER_WEB_PORT": port}), patch("project.web.__main__.load_dotenv"), patch("uvicorn.run") as run:
                with self.assertRaisesRegex(RuntimeError, "CAREER_WEB_PORT"):
                    main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
