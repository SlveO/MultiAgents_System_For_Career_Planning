"""CUDA-only policy for local models; safe to import without torch."""
from __future__ import annotations

import re


def require_cuda(device: str, torch_module=None) -> str:
    if not re.fullmatch(r"cuda(?::\d+)?", device):
        raise RuntimeError("Local models require a CUDA device; CPU fallback is disabled")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("Install server CUDA PyTorch before loading local models") from exc
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local models will not be loaded on CPU")
    torch_module.cuda.get_device_properties(device)
    return device
