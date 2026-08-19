try:
    from ..core.memory_manager import get_vram_manager
except ImportError:
    from project.core.memory_manager import get_vram_manager


class ImageProcessor:
    def __init__(
        self,
        model_path='./models/Qwen3-VL-2B-Instruct',
        *,
        torch_module=None,
        model_class=None,
        processor_class=None,
        vision_info_fn=None,
        vram_manager=None,
    ):
        self.model_path = model_path
        self.vram_manager = vram_manager or get_vram_manager()
        self._torch = torch_module
        self._model_class = model_class
        self._processor_class = processor_class
        self._process_vision_info = vision_info_fn

    def _ensure_dependencies(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    '图像功能需要 GPU profile 中的 torch、transformers 和 qwen-vl-utils。'
                ) from exc
            self._torch = torch

        if self._model_class is None or self._processor_class is None:
            try:
                from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
            except ImportError as exc:
                raise RuntimeError(
                    '图像功能需要 GPU profile 中的 torch、transformers 和 qwen-vl-utils。'
                ) from exc
            self._model_class = Qwen3VLForConditionalGeneration
            self._processor_class = Qwen3VLProcessor

        if self._process_vision_info is None:
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise RuntimeError(
                    '图像功能需要 GPU profile 中的 torch、transformers 和 qwen-vl-utils。'
                ) from exc
            self._process_vision_info = process_vision_info

    def _load(self):
        self._ensure_dependencies()
        device = 'cuda' if self._torch.cuda.is_available() else 'cpu'
        dtype = self._torch.float16 if device == 'cuda' else self._torch.float32
        print('  加载视觉处理器...')
        processor = self._processor_class.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        print(f'  加载视觉模型到{device.upper()}...')
        model = self._model_class.from_pretrained(
            self.model_path,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

        if device == 'cuda':
            self._torch.cuda.synchronize()
        return model, processor, None

    def analyze(self, image_path, question=None, context=None):
        self.vram_manager.ensure_loaded('vision', self._load)
        model_data = self.vram_manager.loaded_models['vision']
        model = model_data['model']
        processor = model_data['processor']

        if question:
            prompt = question
        elif context:
            prompt = f'结合以下背景信息分析图片：\n{context}\n\n请详细描述图片与职业规划相关的信息。'
        else:
            prompt = '请详细描述这张图片内容，并提取对职业规划有帮助的信息。'

        messages = [
            {
                'role': 'system',
                'content': '你是专业图像分析助手，请给出可验证的图像事实。',
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path},
                    {'type': 'text', 'text': prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = self._process_vision_info(messages)
        device = 'cuda' if self._torch.cuda.is_available() else 'cpu'
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to(device)

        with self._torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        return output_text

    def unload(self):
        self.vram_manager.unload_model('vision')
