from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.agents.perception.audio_agent import AudioPerceptionAgent
from project.agents.perception.document_agent import DocumentPerceptionAgent
from project.agents.perception.image_agent import ImagePerceptionAgent
from project.agents.perception.text_agent import TextPerceptionAgent
from project.agents.perception.video_agent import VideoPerceptionAgent


class TestPerceptionAgents(unittest.TestCase):
    def test_chinese_text_extracts_skills_interests_and_weaknesses(self) -> None:
        result = TextPerceptionAgent().perceive(
            "我熟悉 Python 和 SQL，想做数据分析，但英语比较薄弱。"
        )

        serialized = "\n".join(result.facts)
        self.assertIn("Python", serialized)
        self.assertIn("SQL", serialized)
        self.assertIn("数据分析", serialized)
        self.assertIn("薄弱", serialized)

    def test_txt_document_extracts_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.txt"
            path.write_text("熟悉 Python 和 SQL，希望从事数据分析。", encoding="utf-8")
            result = DocumentPerceptionAgent().perceive(str(path))

        self.assertEqual(result.summary, "Parsed profile.txt")
        self.assertIn("Python", result.raw_output)

    def test_docx_document_extracts_content(self) -> None:
        import docx

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.docx"
            document = docx.Document()
            document.add_paragraph("熟悉 Python，希望从事后端开发。")
            document.save(path)
            result = DocumentPerceptionAgent().perceive(str(path))

        self.assertEqual(result.summary, "Parsed profile.docx")
        self.assertIn("后端开发", result.raw_output)

    def test_pdf_document_extracts_page_text(self) -> None:
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            content = DecodedStreamObject()
            content.set_data(b"BT /F1 12 Tf 72 720 Td (Python SQL data analyst) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(content)
            with path.open("wb") as handle:
                writer.write(handle)

            result = DocumentPerceptionAgent().perceive(str(path))

        self.assertEqual(result.summary, "Parsed profile.pdf")
        self.assertIn("Python SQL data analyst", result.raw_output)

    def test_image_agent_can_use_an_injected_lightweight_processor(self) -> None:
        class FakeProcessor:
            def __init__(self, model_path):
                self.model_path = model_path

            def analyze(self, image_path, question=None):
                return "图片展示了一份包含 Python 项目的简历。"

            def unload(self):
                return None

        with patch(
            "project.agents.perception.image_agent._load_image_processor",
            return_value=FakeProcessor,
        ):
            result = ImagePerceptionAgent("unused").perceive(
                "resume.png", user_goal="求职", user_text=""
            )

        self.assertEqual(result.modality, "image")
        self.assertIn("Python", result.raw_output)

    def test_audio_and_video_agents_fail_readably_for_missing_files(self) -> None:
        audio = AudioPerceptionAgent().perceive("missing.wav")
        video = VideoPerceptionAgent().perceive("missing.mp4")

        self.assertEqual(audio.confidence, 0.0)
        self.assertIn("not found", audio.summary.lower())
        self.assertEqual(video.confidence, 0.0)
        self.assertIn("not found", video.summary.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
