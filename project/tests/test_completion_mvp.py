from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from project.agents.perception.document_agent import DocumentPerceptionAgent
from project.core.brain_client import DeepSeekBrainClient
from project.core.input_router import InputClassifier
from project.core.settings import AppSettings, get_settings


class TestCompletionMvpFoundation(unittest.TestCase):
    def test_deepseek_defaults_to_v4_flash_with_thinking_disabled(self) -> None:
        settings = AppSettings(_env_file=None)
        self.assertEqual(settings.deepseek_base_url, "https://api.deepseek.com")
        self.assertEqual(settings.brain_default_model, "deepseek-v4-flash")

        client = DeepSeekBrainClient.__new__(DeepSeekBrainClient)
        client.default_model = settings.brain_default_model
        payload = client._payload("规划职业路线", model=None, stream=False)

        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_get_settings_reuses_one_process_configuration(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            first = get_settings()
            second = get_settings()

        self.assertEqual(id(first), id(second))
        self.assertEqual(first.jwt_secret_key, second.jwt_secret_key)

    def test_document_router_only_claims_formats_the_parser_supports(self) -> None:
        supported = InputClassifier.DOCUMENT_EXTENSIONS

        self.assertEqual(
            supported,
            {".txt", ".md", ".csv", ".tsv", ".pdf", ".docx", ".xlsx"},
        )

    def test_xls_returns_an_unsupported_format_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.xls"
            path.write_bytes(b"not a real workbook")

            result = DocumentPerceptionAgent().perceive(str(path))

        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Unsupported document format", result.summary)

    def test_orchestrator_does_not_initialize_optional_image_stack(self) -> None:
        from project.agents.perception import image_agent
        from project.orchestrator import CareerOrchestrator

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "sessions.db")
            with patch.object(
                image_agent,
                "_load_image_processor",
                side_effect=ImportError("torch is unavailable"),
            ):
                orchestrator = CareerOrchestrator(db_path=db_path)

        self.assertIsNone(orchestrator.image_agent)
        self.assertIsNone(orchestrator.audio_agent)
        self.assertIsNone(orchestrator.video_agent)

    def test_cloud_roadmap_requires_all_three_completion_periods(self) -> None:
        from project.orchestrator import CareerOrchestrator

        complete = [
            {"period": "30d", "objective": "盘点"},
            {"period": "90d", "objective": "项目"},
            {"period": "180d", "objective": "投递"},
        ]
        incomplete = [{"period": "30d", "objective": "盘点"}]

        self.assertTrue(CareerOrchestrator._valid_roadmap(complete))
        self.assertFalse(CareerOrchestrator._valid_roadmap(incomplete))

    def test_image_processor_loads_on_cpu_when_cuda_is_unavailable(self) -> None:
        from project.agents import image as image_module

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def synchronize() -> None:
                raise AssertionError("CPU loading must not synchronize CUDA")

        class FakeTorch:
            cuda = FakeCuda()
            float16 = "float16"
            float32 = "float32"

        class FakeModel:
            def __init__(self) -> None:
                self.device = None

            def to(self, device: str):
                self.device = device
                return self

            def eval(self):
                return self

        fake_model = FakeModel()
        fake_processor = object()

        class FakeProcessorClass:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return fake_processor

        class FakeModelClass:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                return fake_model

        processor = image_module.ImageProcessor(
            model_path="unused",
            torch_module=FakeTorch(),
            model_class=FakeModelClass,
            processor_class=FakeProcessorClass,
            vision_info_fn=lambda _messages: ([], []),
            vram_manager=object(),
        )
        loaded_model, loaded_processor, _ = processor._load()

        self.assertIs(loaded_model, fake_model)
        self.assertIs(loaded_processor, fake_processor)
        self.assertEqual(fake_model.device, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
