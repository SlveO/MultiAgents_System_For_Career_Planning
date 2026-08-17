"""最简调用示例：展示如何用 src/model 的代码调用模型。

用法（仓库根目录）：
  .venv/Scripts/python.exe scripts/try_chat.py
"""

import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 第1步：把 .env 里的密钥灌进环境变量（客户端只认环境变量，不读文件）
env_file = ROOT / ".env"
for line in open(env_file, encoding="utf-8"):
    if "=" in line and not line.startswith("#"):
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# 第2步：创建客户端（密钥从环境变量读取）
from src.model.deepseek_client import DeepSeekClient

client = DeepSeekClient()

# 第3步：一次对话调用
content, usage = client.chat(
    [{"role": "user", "content": "用一句话介绍你自己"}],
    json_mode=False,  # 自由对话不用 JSON 模式
)
print("模型回复:", content)
print("消耗 tokens:", usage.get("total_tokens"))

# 第4步：也可以调画像提取（另一个入口）
from src.model.model_adapter import ModelAdapter

result = ModelAdapter(client).extract_profile("我是计算机专业大三学生，会Python")
print("画像:", result["profile"])
