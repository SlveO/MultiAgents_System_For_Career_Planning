"""第一周真实验证：/models 确认 + 3 个样例画像提取。

用法（项目根目录）：
  .venv/Scripts/python.exe scripts/demo_profile.py
环境变量 DEEPSEEK_API_KEY 缺失时会自动读取本地 .env（仅脚本层便利，客户端本身只读环境变量）。
"""

import json
import os
import sys
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码（演示/截图需要）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

from src.model.deepseek_client import DeepSeekClient  # noqa: E402
from src.model.model_adapter import ModelAdapter  # noqa: E402


def main():
    client = DeepSeekClient()
    models = client.list_models()
    print(f"[1/2] /models 可用模型列表：{models}")
    if client.model not in models:
        print(f"[1/2] 警告：{client.model} 不在 /models 列表中，仍继续尝试调用")
    else:
        print(f"[1/2] 确认 {client.model} 可用")

    adapter = ModelAdapter(client)
    samples = json.loads(
        (ROOT / "tests" / "fixtures" / "profile_samples.json").read_text(encoding="utf-8")
    )
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n=== 样例 {i}/{len(samples)}：{sample['id']} ===")
        print(f"输入：{sample['input']}")
        result = adapter.extract_profile(sample["input"])
        result["sample_id"] = sample["id"]
        result["input"] = sample["input"]
        results.append(result)
        print(f"used_fallback={result['used_fallback']} elapsed_ms={result['elapsed_ms']}")
        print(f"token用量={result['usage']}")
        print("画像：")
        print(json.dumps(result["profile"], ensure_ascii=False, indent=2))
        if result["errors"]:
            print("错误：")
            print(json.dumps(result["errors"], ensure_ascii=False, indent=2))

    out_dir = ROOT / "outputs" / "week1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "profile_baseline_results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if not r["used_fallback"])
    print(f"\n[2/2] 结果已保存：{out_file}")
    print(f"[2/2] 汇总：{ok}/{len(results)} 个样例由 API 成功提取（其余走规则兜底）")


if __name__ == "__main__":
    main()
