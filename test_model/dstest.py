# Calling Deepseek by Transformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_path = './models/DeepSeek-R1-Distill-Qwen-1___5B'


# Loading the model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,  # 修复：torch_dtype -> dtype
    device_map='auto',
    trust_remote_code=True
)

# Loading the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# Pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Input preparation
messages = [
    {
        "role": "system",
        "content": "你是一个人工智能助手，协助用户解答问题和提供信息。",
    },
    {
        "role": "user",
        "content": "介绍你自己,并讲解什么是神经网络。",
    }
]

# Applying the template
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 修复：添加attention_mask
input_ids = tokenizer(
    input_text,
    return_tensors='pt'
).to(model.device)

# Streaming generation setup
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=False
)

gen_kwargs = {
    "input_ids": input_ids.input_ids,
    "attention_mask": input_ids.attention_mask,  # 修复：添加attention_mask
    "streamer": streamer,
    "max_new_tokens": 2048,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "pad_token_id": tokenizer.eos_token_id,
}

# Generating the answer
Thread(
    target=model.generate,
    kwargs=gen_kwargs
).start()

print("=" * 40)
print("Thinking:")
print("-" * 40)

full_text = ""
thinking = ""
response = ""
in_thinking = True

for token in streamer:
    full_text += token

    if in_thinking:
        # 检测  结束思考
        if " 在思考阶段检测结束标签，分离思考内容和回答内容。 \n" in full_text:
            # 提取思考内容（从开始到  之前）
            think_end = full_text.find(" \n")
            thinking = full_text[:think_end].replace(" \n", "").strip()
            
            # 提取回答内容（  之后）
            answer_start = full_text[think_end + 11:]  # 11是" \n"的长度
            
            print(thinking)  # 打印完整思考内容
            print("-" * 40)
            print("回答:", end="", flush=True)
            
            # 输出并累加回答内容
            if answer_start:
                print(answer_start, end="", flush=True)
                response = answer_start
            
            in_thinking = False
        else:
            # 还在思考中，实时输出
            print(token, end="", flush=True)
    else:
        # 回答阶段，直接输出并累加
        print(token, end="", flush=True)
        response += token

print("\n" + "=" * 40)

# 兜底解析（如果上面没成功分离）
if not thinking and " \n" in full_text:
    parts = full_text.split(" \n", 1)
    thinking = parts[0].replace(" \n", "").strip()
    response = parts[1].strip() if len(parts) > 1 else full_text.strip()

print(f"\n📊 思考:{len(thinking)}字 | 回答:{len(response)}字")

# 可选：分别打印思考和回答
print("\n【完整思考内容】")
print(thinking)
print("\n【完整回答内容】")
print(response)