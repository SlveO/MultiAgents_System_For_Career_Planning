# text_processor.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

try:
    from ..core.memory_manager import get_vram_manager
except ImportError:
    from project.core.memory_manager import get_vram_manager


class TextProcessor:
    def __init__(self, model_path='./models/DeepSeek-R1-Distill-Qwen-1.5B'):
        self.model_path = model_path
        self.vram_manager = get_vram_manager()

    def _load(self):
        """返回 (model, processor, tokenizer)"""
        print('  加载分词器...')
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print('  加载模型到GPU...')
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            trust_remote_code=True,
        ).cuda().eval()

        torch.cuda.synchronize()
        return model, None, tokenizer

    def generate(self, query, system_prompt=None, stream=False, max_tokens=2048):
        self.vram_manager.ensure_loaded('text', self._load)
        model_data = self.vram_manager.loaded_models['text']
        model = model_data['model']
        tokenizer = model_data['tokenizer']

        if system_prompt is None:
            system_prompt = '你是一个职业规划助手，请基于用户信息给出清晰、可执行的建议。'

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt').to('cuda')

        if stream:
            return self._stream_generate(model, tokenizer, inputs, max_tokens)
        return self._batch_generate(model, tokenizer, inputs, max_tokens)

    def _batch_generate(self, model, tokenizer, inputs, max_tokens):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        thinking, response = self._parse_thinking(generated_text)
        return {'thinking': thinking, 'response': response}

    def _stream_generate(self, model, tokenizer, inputs, max_tokens):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'streamer': streamer,
            'max_new_tokens': max_tokens,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.95,
            'pad_token_id': tokenizer.eos_token_id,
        }
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        return self._stream_parser(streamer)

    def _stream_parser(self, streamer):
        full_text = ''
        thinking = ''
        response = ''
        in_thinking = True
        split_tag = '</think>'

        for token in streamer:
            full_text += token
            if in_thinking:
                if split_tag in full_text:
                    think_end = full_text.find(split_tag)
                    thinking = full_text[:think_end].replace('<think>', '').strip()
                    answer_start = full_text[think_end + len(split_tag):]
                    yield {'type': 'thinking_complete', 'thinking': thinking, 'token': answer_start}
                    response = answer_start
                    in_thinking = False
                else:
                    yield {'type': 'thinking', 'token': token}
            else:
                response += token
                yield {'type': 'response', 'token': token}

        if not thinking and split_tag in full_text:
            parts = full_text.split(split_tag, 1)
            thinking = parts[0].replace('<think>', '').strip()
            response = parts[1].strip() if len(parts) > 1 else full_text.strip()

        yield {'type': 'complete', 'thinking': thinking, 'response': response}

    def _parse_thinking(self, text):
        split_tag = '</think>'
        if split_tag in text:
            parts = text.split(split_tag, 1)
            thinking = parts[0].replace('<think>', '').strip()
            response = parts[1].strip()
            return thinking, response
        return '', text.strip()

    def unload(self):
        self.vram_manager.unload_model('text')
