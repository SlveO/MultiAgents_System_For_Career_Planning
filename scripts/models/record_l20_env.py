"""Capture a sanitized technical record for the current L20 server session."""
from __future__ import annotations

import getpass
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from project.core.settings import get_settings  # noqa: E402

DEFAULT_MODELS_DIR = Path(get_settings().vision_model_path).parent

_HOSTNAME = socket.gethostname()
_USERNAME = getpass.getuser()
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{12,}")
_HOME_PATTERN = re.compile(r"(/home/|/Users/)[A-Za-z0-9_.-]+")


def redact(text: str) -> str:
    if _HOSTNAME:
        text = text.replace(_HOSTNAME, "[REDACTED_HOST]")
    text = _HOME_PATTERN.sub(r"\1[REDACTED_USER]", text)
    if _USERNAME:
        text = text.replace(_USERNAME, "[REDACTED_USER]")
    return _KEY_PATTERN.sub("[REDACTED_KEY]", text)


def run_cmd(args: List[str], timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return f"command timed out: {args[0]}"
    output = (completed.stdout or "").strip()
    if completed.returncode:
        output = f"{output}\n(exit code {completed.returncode}: {(completed.stderr or '').strip()})"
    return redact(output)


def os_release() -> str:
    try:
        pairs = {}
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                pairs[key] = value.strip().strip('"')
        return f"{pairs.get('PRETTY_NAME', '?')} (codename {pairs.get('VERSION_CODENAME', '?')})"
    except OSError as exc:
        return f"unable to read /etc/os-release: {exc}"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def torch_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "version": "not installed",
            "cuda_runtime": None,
            "cuda_available": False,
            "bf16_supported": False,
        }
    available = torch.cuda.is_available()
    return {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
    }


def network_check() -> str:
    try:
        socket.getaddrinfo("www.modelscope.cn", 443)
    except OSError as exc:
        return f"modelscope.cn DNS failed: {exc}"
    return "modelscope.cn DNS resolves; no artifact downloaded"


def collect() -> Dict[str, Any]:
    if platform.system() != "Linux":
        raise SystemExit(
            "this script records the L20 Ubuntu host only; "
            f"current platform is {platform.system()}"
        )
    return {
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": os_release(),
        "kernel": run_cmd(["uname", "-a"]),
        "python": run_cmd([sys.executable, "--version"]),
        "nvidia_smi": run_cmd([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]),
        "cuda_toolkit": run_cmd(["nvcc", "--version"]),
        "torch": torch_info(),
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "modelscope": package_version("modelscope"),
        "pymupdf": package_version("PyMuPDF"),
        "disk": run_cmd(["df", "-B1", str(REPO_ROOT), str(DEFAULT_MODELS_DIR)]),
        "network": network_check(),
    }


def to_markdown(record: Dict[str, Any]) -> str:
    labels = {
        "os": "Ubuntu 版本", "kernel": "内核", "python": "Python",
        "nvidia_smi": "GPU / NVIDIA 驱动 / 显存",
        "cuda_toolkit": "本地 CUDA Toolkit", "torch": "PyTorch / CUDA / BF16",
        "torchvision": "torchvision", "transformers": "Transformers",
        "modelscope": "ModelScope", "pymupdf": "PyMuPDF",
        "disk": "磁盘", "network": "网络",
    }
    lines = [
        f"Captured at (UTC): {record['captured_at_utc']}",
        "",
        "| 项目 | 记录 |",
        "|---|---|",
    ]
    for key, label in labels.items():
        value = (
            json.dumps(record[key], ensure_ascii=False)
            if isinstance(record[key], dict)
            else str(record[key])
        )
        lines.append(f"| {label} | {value.replace(chr(10), '<br>')} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Record a sanitized L20 software environment.")
    parser.parse_args(argv)
    record = collect()
    print(to_markdown(record))
    output_dir = REPO_ROOT / "data/experiments/environment"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"l20-env-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrecord written to {output.relative_to(REPO_ROOT)} (ignored by git)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
