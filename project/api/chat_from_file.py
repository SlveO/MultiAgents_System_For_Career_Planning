from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Standard multimodal chat client (file-based input).")
    parser.add_argument("--file", default="request.json", help="Path to request json file.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/multimodal/chat/stream",
        help="Streaming API endpoint.",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds.")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"request file not found: {file_path}")

    # Support both UTF-8 and UTF-8 BOM encoded JSON files.
    payload = json.loads(file_path.read_text(encoding="utf-8-sig"))

    with httpx.stream("POST", args.url, json=payload, timeout=args.timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            print(line)


if __name__ == "__main__":
    main()
