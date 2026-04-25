# image_processor.py
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

try:
    from ..core.memory_manager import get_vram_manager
except ImportError:
    from project.core.memory_manager import get_vram_manager


class ImageProcessor:
    def __init__(self, model_path='./models/Qwen3-VL-2B-Instruct'):
        self.model_path = model_path
        self.vram_manager = get_vram_manager()

    def _load(self):
        print('  加载视觉处理器...')
        processor = Qwen3VLProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        print('  加载视觉模型到GPU...')
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            trust_remote_code=True,
        ).cuda().eval()

        torch.cuda.synchronize()
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

        if process_vision_info is None:
            raise RuntimeError('缺少 qwen_vl_utils.process_vision_info，请安装相关依赖后再使用图像功能。')

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to('cuda')

        with torch.no_grad():
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
