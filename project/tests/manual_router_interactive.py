from __future__ import annotations

import argparse
import json
from typing import Any, Dict

try:
    from project.core.input_router import DataRouter, InputClassifier
except ImportError:
    from core.input_router import DataRouter, InputClassifier


def _print_result(classification: Dict[str, Any], routed: Dict[str, Any], valid: bool, message: str) -> None:
    print("\n" + "=" * 72)
    print(f"mode: {classification.get('mode')} | valid: {valid} | msg: {message}")
    print("-" * 72)
    print("[classification]")
    print(json.dumps(classification, ensure_ascii=False, indent=2))
    print("-" * 72)
    print("[route]")
    print(json.dumps(routed, ensure_ascii=False, indent=2))
    print("=" * 72 + "\n")


def interactive_loop(strict: bool = False) -> None:
    classifier = InputClassifier()
    router = DataRouter()

    print("Unified Router Interactive Tester")
    print("Commands: :help  :quit  :examples")
    print("Tips:")
    print('  1) 直接粘贴真实文件路径（支持带空格，建议加引号）')
    print('  2) 图文混合示例：请分析这张图 "D:\\data\\a b\\img.jpg"')
    print("  3) 文档支持：doc/docx/pdf/xls/xlsx/csv/tsv/ods/txt/md")
    print()

    while True:
        user_input = input("input> ").strip()
        if not user_input:
            continue

        if user_input.lower() in {":quit", "quit", "exit", "q"}:
            print("bye")
            return

        if user_input.lower() == ":help":
            print("输入任意文本/路径进行测试；可混合输入文本 + 图片路径。")
            print("非 text 模式会检查文件是否存在。")
            continue

        if user_input.lower() == ":examples":
            print('示例1: "D:\\real\\photo.jpg"')
            print('示例2: "D:\\real\\resume.docx"')
            print('示例3: "D:\\real\\report.pdf"')
            print('示例4: "D:\\real\\sheet.xlsx"')
            print('示例5: "D:\\real\\voice.mp3"')
            print('示例6: "D:\\real\\video.mp4"')
            print('示例7: 请总结这个图 "D:\\real\\diagram.png"')
            continue

        classification = classifier.classify(user_input)
        valid, message = classifier.validate(classification)
        routed = router.route(user_input)

        if strict and not valid:
            print(f"[STRICT] invalid input: {message}")
            continue

        _print_result(classification, routed, valid, message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual interactive tester for unified input router.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="when set, invalid inputs are rejected before printing route payload",
    )
    args = parser.parse_args()
    interactive_loop(strict=args.strict)


if __name__ == "__main__":
    main()

