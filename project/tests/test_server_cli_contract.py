from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from project import assistant_cli
from project.agents.image import ImageProcessor
from project.agents.perception.audio_agent import AudioPerceptionAgent
from project.core.gpu import require_cuda


class TestServerCliContract(unittest.TestCase):
    def test_contract_matches_confirmed_scope(self):
        root = Path(__file__).resolve().parents[2]
        contract = json.loads((root / "dataset/completion_protocol.json").read_text())
        self.assertEqual(contract["entrypoint"], "project.assistant_cli")
        self.assertEqual(contract["reasoning_backend"], "deepseek-api")
        self.assertEqual(contract["other_model_backend"], "local-cuda")
        self.assertFalse(contract["allow_cpu_model_loading"])
        self.assertFalse(contract["architecture_comparison_enabled"])
        self.assertFalse(contract["accept_other_unmerged_branches"])

    def test_cpu_device_rejected_without_importing_model(self):
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            require_cuda("cpu")
        with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            AudioPerceptionAgent(device="cpu")._lazy_load()

    def test_missing_cuda_rejected_before_loading_weights(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        model = MagicMock()
        proc = ImageProcessor(torch_module=torch, model_class=model)
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            proc._load()
        model.from_pretrained.assert_not_called()

    def test_local_vision_uses_selected_cuda_and_local_files_only(self):
        torch, model_class, processor_class = MagicMock(), MagicMock(), MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        with tempfile.TemporaryDirectory() as tmp:
            proc = ImageProcessor(tmp, device="cuda:1", torch_module=torch,
                                  model_class=model_class, processor_class=processor_class)
            proc._load()
        kwargs = model_class.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": "cuda:1"})
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(processor_class.from_pretrained.call_args.kwargs["local_files_only"])
        model_class.from_pretrained.return_value.to.assert_not_called()

    def test_cli_does_not_report_template_as_success(self):
        orchestrator = MagicMock()
        orchestrator.run.return_value.model_dump.return_value = {"served_by": "local_fallback"}
        inputs = MagicMock(side_effect=AssertionError("no feedback on failed plan"))
        output = []
        code = assistant_cli.main(["--goal", "career", "--no-follow-up"],
                                  orchestrator_factory=lambda: orchestrator,
                                  input_fn=inputs, output_fn=output.append)
        self.assertEqual(code, 1)
        orchestrator.submit_feedback.assert_not_called()

    def test_cli_stream_failure_is_not_success(self):
        orchestrator = MagicMock()
        orchestrator.run_stream.return_value = iter([])
        code = assistant_cli.main(["--goal", "career", "--no-follow-up", "--stream"],
                                  orchestrator_factory=lambda: orchestrator,
                                  output_fn=lambda _: None)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
