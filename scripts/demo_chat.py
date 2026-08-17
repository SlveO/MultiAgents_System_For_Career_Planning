"""与 deepseek-v4-flash 的真实对话演示（消耗真实余额）。

用法（项目根目录）：
  .venv/Scripts/python.exe scripts/demo_chat.py
输入你的话开始多轮对话，输入 exit 退出。
"""

import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码；
# 管道输入（非交互）同样按 UTF-8 解码，避免中文变乱码导致 API 400
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure") and not sys.stdin.isatty():
    sys.stdin.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not os.environ.get("DEEPSEEK_API_KEY"):
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from src.model.deepseek_client import DeepSeekClient, ModelError  # noqa: E402


def main():
    client = DeepSeekClient()
    print(f"已连接 {client.model}（{client.base_url}）")
    print("开始对话，输入 exit 退出。每次回复消耗少量余额。\n")
    history = []
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("已退出。")
            break
        history.append({"role": "user", "content": user_input})
        try:
            content, usage = client.chat(history, json_mode=False, temperature=0.7)
        except ModelError as exc:
            err = exc.to_error_dict()
            print(f"[{err['code']}] {err['message']}")
            history.pop()  # 回滚这条失败消息，保持上下文干净
            continue
        history.append({"role": "assistant", "content": content})
        print(f"\n助手: {content}\n")
        print(f"(本次消耗 {usage.get('total_tokens', '?')} tokens)\n")


if __name__ == "__main__":
    main()
