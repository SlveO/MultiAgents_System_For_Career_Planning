from __future__ import annotations

import unittest

from project.assistant_cli import _display_version


class TestCliReadingView(unittest.TestCase):
    def test_saved_projection_is_displayed_with_counts_and_notice(self):
        output = []
        _display_version({"display_text": "完整正文🙂", "reading_view": {
            "mode": "brief", "excerpt": "摘选🙂", "notice": "保留原文",
        }}, output.append)
        self.assertEqual(output, ["简要正文 3 字符；完整正文 5 字符。", "摘选🙂", "保留原文"])

    def test_legacy_version_remains_readable(self):
        output = []
        _display_version({"display_text": "旧版正文"}, output.append)
        self.assertEqual(output, ["旧版正文"])


if __name__ == "__main__":
    unittest.main()
