from __future__ import annotations

import ast
import unittest
from pathlib import Path


class TestMainEntryAst(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parents[1] / "main.py"
        self.tree = ast.parse(self.path.read_text(encoding="utf-8"))

    def test_legacy_module_delegates_to_canonical_assistant_cli(self) -> None:
        imports_main = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "assistant_cli"
            and any(alias.name == "main" for alias in node.names)
            for node in self.tree.body
        )
        self.assertTrue(imports_main)

    def test_legacy_module_contains_no_duplicate_assistant_class(self) -> None:
        class_names = {
            node.name for node in self.tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertNotIn("MultimodalAssistant", class_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
