from __future__ import annotations

import unittest

from project.core.input_router import DataRouter, InputClassifier


class TestInputRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = InputClassifier()
        self.router = DataRouter()
        self.image = "./samples/sample.jpg"
        self.word = "./samples/sample.docx"
        self.pdf = "./samples/sample.pdf"
        self.excel = "./samples/sample.xlsx"
        self.audio = "./samples/sample.mp3"
        self.video = "./samples/sample.mp4"

    def test_text_classification_and_route(self) -> None:
        result = self.classifier.classify("this is plain text")
        self.assertEqual(result["mode"], "text")

        routed = self.router.route("this is plain text")
        self.assertEqual(routed["route"], "text")
        self.assertEqual(routed["target_agent"], "agents.perception.TextPerceptionAgent")

    def test_image_classification_and_route(self) -> None:
        result = self.classifier.classify(self.image)
        self.assertEqual(result["mode"], "image")

        routed = self.router.route(self.image)
        self.assertEqual(routed["route"], "image")
        self.assertEqual(routed["target_agent"], "agents.perception.ImagePerceptionAgent")

    def test_multimodal_text_image_route(self) -> None:
        text = f'please analyze "{self.image}"'
        result = self.classifier.classify(text)
        self.assertEqual(result["mode"], "multimodal")
        self.assertTrue(result["text_content"].startswith("please analyze"))

        routed = self.router.route(text)
        self.assertEqual(routed["route"], "multimodal")
        self.assertEqual(routed["target_agent"], "agents.perception.ImagePerceptionAgent")

    def test_document_file_route_word_pdf_excel(self) -> None:
        for path in (self.word, self.pdf, self.excel):
            with self.subTest(path=path):
                result = self.classifier.classify(path)
                self.assertEqual(result["mode"], "file")

                routed = self.router.route(path)
                self.assertEqual(routed["route"], "file")
                self.assertEqual(routed["target_agent"], "agents.perception.DocumentPerceptionAgent")

    def test_audio_video_route(self) -> None:
        for path in (self.audio, self.video):
            with self.subTest(path=path):
                result = self.classifier.classify(path)
                self.assertEqual(result["mode"], "audio_video")

                routed = self.router.route(path)
                self.assertEqual(routed["route"], "audio_video")
                self.assertEqual(routed["target_agent"], "agents.perception.AudioPerceptionAgent")


if __name__ == "__main__":
    unittest.main(verbosity=2)

