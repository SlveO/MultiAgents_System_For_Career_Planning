"""Capture a sanitized L20 environment record for docs/l20-setup.md.

Day-1 deliverable for member B: run this on the L20 Ubuntu host BEFORE
installing anything. It collects OS, kernel, driver, GPU, CUDA, Python,
PyTorch, disk, network, and VRAM state, then redacts hostname, username,
private paths, and key-shaped tokens before printing a markdown block.

The same record is written as JSON to the ignored `data/` directory so the
raw (already redacted) evidence stays reproducible. Never dump environment
variables or `nvidia-smi` process tables (both can contain accounts).
"""

from __future__ import annotations

import getpass
import json
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ALLOWED_PLATFORM = "Linux"

_HOSTNAME = socket.gethostname()
_USERNAME = getpass.getuser()
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{12,}")
# Any home-directory path may identify someone; redact the user segment
# regardless of whether it matches the account running the capture.
_HOME_PATTERN = re.compile(r"(/home/|/Users/)[A-Za-z0-9_.-]+")


def redact(text: str) -> str:
    """Strip identity-bearing content from captured command output."""
    if _HOSTNAME:
        text = text.replace(_HOSTNAME, "[REDACTED_HOST]")
    text = _HOME_PATTERN.sub(r"\1[REDACTED_USER]", text)
    if _USERNAME:
        text = text.replace(_USERNAME, "[REDACTED_USER]")
    return _KEY_PATTERN.sub("[REDACTED_KEY]", text)


def run_cmd(args: List[str], timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout
        )
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            out = f"{out}\n(exit code {proc.returncode}: {(proc.stderr or '').strip()})"
        return redact(out)
    except FileNotFoundError:
        return redact(f"command not found: {args[0]}")
    except subprocess.TimeoutExpired:
        return redact(f"command timed out: {args[0]}")


def os_release() -> str:
    """PRETTY_NAME and VERSION_CODENAME only; the full file may carry more."""
    try:
        pairs = {}
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                pairs[key] = value.strip().strip('"')
        return f"{pairs.get('PRETTY_NAME', '?')} (codename {pairs.get('VERSION_CODENAME', '?')})"
    except OSError as exc:
        return f"unable to read /etc/os-release: {exc}"


def torch_info() -> str:
    try:
        import torch  # lazy GPU import
    except ImportError:
        return "torch not installed yet"
    cuda = torch.version.cuda
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        return (
            f"torch {torch.__version__}, CUDA runtime {cuda}, "
            f"cuda.is_available()=True, device0={device}"
        )
    return f"torch {torch.__version__}, CUDA runtime {cuda}, cuda.is_available()=False"


def network_check() -> str:
    try:
        socket.getaddrinfo("www.modelscope.cn", 443)
        resolved = True
    except OSError as exc:
        return f"modelscope.cn DNS failed: {exc}"
    try:
        import urllib.request

        with urllib.request.urlopen(
            "https://www.modelscope.cn", timeout=10
        ) as resp:
            return f"modelscope.cn resolves and answers HTTPS (HTTP {resp.status})"
    except Exception as exc:  # network check must not crash the record
        return f"modelscope.cn resolves but HTTPS check failed: {exc}"


def collect() -> Dict[str, str]:
    if platform.system() != ALLOWED_PLATFORM:
        raise SystemExit(
            "this script records the L20 Ubuntu host only; "
            f"current platform is {platform.system()}"
        )
    return {
        "os": os_release(),
        "kernel": run_cmd(["uname", "-a"]),
        "python": run_cmd([sys.executable, "--version"]).strip()
        or f"Python {platform.python_version()}",
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv",
            ]
        ),
        "torch": torch_info(),
        "disk": run_cmd(["df", "-h", str(REPO_ROOT), str(REPO_ROOT / "models")])
        if (REPO_ROOT / "models").exists()
        else run_cmd(["df", "-h", str(REPO_ROOT)]),
        "network": network_check(),
    }


def to_markdown(record: Dict[str, str]) -> str:
    lines = [
        f"Captured at (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
        "",
        "| 项目 | 记录 |",
        "|---|---|",
    ]
    labels = {
        "os": "Ubuntu 版本",
        "kernel": "内核 (uname -a, 主机名已脱敏)",
        "python": "Python",
        "nvidia_smi": "驱动 / 显存 (nvidia-smi)",
        "torch": "PyTorch / CUDA",
        "disk": "磁盘 (df -h, 私人路径已脱敏)",
        "network": "网络 (modelscope.cn)",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {record[key]} |")
    return "\n".join(lines)


def main() -> int:
    record = collect()
    print(to_markdown(record))

    out_dir = REPO_ROOT / "data" / "l20"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_file = out_dir / f"l20-env-{stamp}.json"
    out_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nrecord written to {out_file.relative_to(REPO_ROOT)} (ignored by git)")
    print("paste the table above into docs/l20-setup.md (replace placeholders).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
