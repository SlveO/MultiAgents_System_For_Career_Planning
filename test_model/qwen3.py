import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

# Loading the model
model_path = './models/Qwen3-VL-2B-Instruct'

# Initialize the model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
).eval()

# Initialize the processor
processor = Qwen3VLProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Input Messages
messages = [    
    {
        "role": "system",
        "content": "你是一个专业图片识别助手，能够分析图片内容并提供详细描述，输出给下一级的语言大模型，使语言大模型准确理解图片内容。"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./project/image/test.jpg"
            },
            {
                "type": "text",
                "text": "请分析这张图片的内容，并提供详细描述。判断图片中的人物是谁并给出判断依据。（唐纳德·特朗普）"
            }
        ]
    }
]


# 应用模板
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 处理视觉信息（变量名修正）
image_inputs, video_inputs = process_vision_info(messages)  # 注意拼写 video 不是 vedio

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,  # 拼写修正
    padding=True,
    return_tensors='pt'
).to(model.device)

# 生成
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

# 解码
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
)[0]

print(output_text)