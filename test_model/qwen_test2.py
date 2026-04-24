# Transformer 


import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor  # 修改为 Qwen3VL
from qwen_vl_utils import process_vision_info

# 加载模型
model_path = './models/Qwen3-VL-2B-Instruct'

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
).eval()

processor = Qwen3VLProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 准备对话（注意：变量名统一为 messages）
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            {
                "type": "text",
                "text": "描述图片内容。"
            },
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
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
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