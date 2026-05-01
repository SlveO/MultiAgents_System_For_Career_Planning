from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TestMainEntryAst(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[1] / "main.py"
        self.tree = ast.parse(self.path.read_text(encoding="utf-8"))

    def test_default_image_model_path(self) -> None:
        source = self.path.read_text(encoding="utf-8")
        self.assertIn("./models/Qwen3-VL-2B-Instruct", source)

    def test_mode_handlers_exist(self) -> None:
        method_names = {
            node.name
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MultimodalAssistant"
            for node in node.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_handle_text_only", method_names)
        self.assertIn("_handle_image_only", method_names)
        self.assertIn("_handle_multimodal", method_names)
        self.assertIn("_handle_file", method_names)
        self.assertIn("_handle_audio_video", method_names)
        self.assertNotIn("_unsupported_mode_response", method_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
