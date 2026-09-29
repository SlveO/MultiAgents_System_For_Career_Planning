"""Local vision inference adapted from server snapshot 7995c62 load/generate."""
from pathlib import Path

from project.core.gpu import require_cuda
from project.core.settings import get_settings


class ImageProcessor:
    def __init__(self, model_path=None, *, device=None, torch_module=None,
                 model_class=None, processor_class=None):
        settings = get_settings()
        self.model_path = model_path or settings.vision_model_path
        self.device = device or settings.local_model_device
        self._torch = torch_module
        self._model_class = model_class
        self._processor_class = processor_class
        self._model = None
        self._processor = None

    def _load(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Vision inference requires the server GPU dependencies") from exc
            self._torch = torch
        require_cuda(self.device, self._torch)
        if not Path(self.model_path).is_dir():
            raise RuntimeError("VISION_MODEL_PATH must point to an existing local model directory")
        with self._torch.cuda.device(self.device):
            if not self._torch.cuda.is_bf16_supported():
                raise RuntimeError("Vision inference requires a BF16-capable CUDA device")
        if self._model_class is None or self._processor_class is None:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            self._model_class = Qwen3VLForConditionalGeneration
            self._processor_class = AutoProcessor
        processor = self._processor_class.from_pretrained(
            self.model_path, local_files_only=True,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
        model = self._model_class.from_pretrained(
            self.model_path, local_files_only=True,
            dtype=self._torch.bfloat16, device_map={"": self.device},
        ).eval()
        return model, processor, None

    def analyze(self, image_path, question=None, context=None):
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError("Image file does not exist")
        if self._model is None:
            self._model, self._processor, _ = self._load()
        prompt = question or "Extract verifiable career-related facts from the image."
        if context:
            prompt += "\n" + context
        messages = [
            {"role": "system", "content": "Report observed facts only; mark missing information explicitly."},
            {"role": "user", "content": [
                {"type": "image", "url": str(path.resolve())},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        trimmed = [out[len(src):] for src, out in zip(inputs.input_ids, generated)]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def unload(self):
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            with self._torch.cuda.device(self.device):
                self._torch.cuda.empty_cache()
