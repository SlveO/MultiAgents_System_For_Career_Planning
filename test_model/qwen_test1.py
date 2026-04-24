import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info  # 关键导入
from threading import Thread

# 加载模型
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     './models/Qwen3-VL-2B-Instruct', 
#     torch_dtype=torch.float16,
#     device_map='auto',
#     trust_remote_code=True
# )

# processor = AutoProcessor.from_pretrained(
#     './models/Qwen3-VL-2B-Instruct',
#     trust_remote_code=True
# )

class QwenChatBot:
    def __init__(self, model_path='./models/Qwen3-VL-2B-Instruct'):
        """初始化模型和处理器"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 对话历史存储
        self.messages = []
        self.max_history = 10  # 保留最近10轮对话

    def add_message(self, role, content):
        """添加消息到历史记录"""
        self.messages.append({"role": role, "content": content})
        # 限制历史长度，防止过长
        if len(self.messages) > self.max_history * 2 + 1:  # system + n轮(user+assistant)
            # 保留 system 和最近的历史
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2):]


    def build_input(self, user_content):
        """构建包含历史的输入"""
        # 添加用户新消息
        temp_messages = self.messages.copy()
        temp_messages.append({"role": "user", "content": user_content})
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理视觉信息（如果有图片）
        image_inputs, video_inputs = process_vision_info(temp_messages)
        
        # 编码输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        return inputs, temp_messages

    def generate_stream(self, user_content, max_new_tokens=512, temperature=0.7):
        """
        流式生成回复
        user_content: 用户输入，可以是纯文本或图文混合列表
        """
        # 构建输入
        inputs, temp_messages = self.build_input(user_content)
        
        # 创建流式输出器
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,      # 跳过输入提示
            skip_special_tokens=True  # 跳过特殊token
        )
        
        # 生成参数
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1
        }
        
        # 在后台线程运行生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 流式输出
        generated_text = ""
        print("Assistant: ", end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)
            generated_text += new_text
        print()  # 换行
        
        thread.join()
        
        # 更新历史记录
        self.add_message("user", user_content)
        self.add_message("assistant", generated_text)
        
        return generated_text

    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        print("对话历史已清空")

    def show_history(self):
        """显示当前对话历史"""
        print("\n=== 当前对话历史 ===")
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # 图文混合内容
                text_parts = [c["text"] for c in content if c["type"] == "text"]
                image_parts = [c["image"] for c in content if c["type"] == "image"]
                print(f"[{role}]: {text_parts}")
                if image_parts:
                    print(f"       [图片: {image_parts}]")
            else:
                print(f"[{role}]: {content}")
        print("===================\n")

def main():
    # 初始化机器人
    print("正在加载模型...")
    bot = QwenChatBot('./models/Qwen3-VL-2B-Instruct')
    print("模型加载完成！\n")
    
    # 设置系统提示（可选）
    bot.messages.append({
        "role": "system",
        "content": """
                
                """
    })
               

    # 示例1：纯文本多轮对话（流式输出）
    print("=== 文本对话模式 ===")
    
    # 第一轮
    print("\nUser: 你好，请介绍一下自己")
    response = bot.generate_stream("你好，请介绍一下自己")
    
    # 第二轮（带历史记忆）
    print("\nUser: 请复述刚才的问题和你的回答")
    response = bot.generate_stream("请复述刚才的问题和你的回答")  # 模型会记住之前的自我介绍
    
    # 第三轮
    print("\nUser: 说一段绕口令")
    response = bot.generate_stream("说一段绕口令")
    
    # 查看历史
    bot.show_history()

if __name__ == "__main__":
    main()


# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {
#                 "type": "text", 
#                 "text": """
#                 
#                 """},
#         ],  
#     }   
# ]

# # 应用模板
# text = processor.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# # 使用独立函数处理视觉信息（不是 processor 的方法）
# image_inputs, video_inputs = process_vision_info(messages)

# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     return_tensors="pt",
#     padding=True
# ).to(model.device)

# # 生成
# generated_ids = model.generate(**inputs, max_new_tokens=128)

# # 解码
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, 
#     skip_special_tokens=True, 
#     clean_up_tokenization_spaces=False
# )

# print(output_text)