"""Frozen local research models and the lazy-GPU adapter that runs them.

This module is member B's deliverable for the controlled architecture
comparison (docs/experiments.md). It must stay importable without `torch` or
`transformers` installed: the core profile and the offline runner import this
module for the registry only, while all heavy dependencies load inside
methods.

Real weight downloads and primary runs are restricted to the networked L20
Ubuntu host; the adapter itself never downloads anything and simply loads
whatever is present under the ignored local `models/` directory.

Registry revisions are ModelScope `master` commit hashes captured on
2026-08-20 with:

    git ls-remote https://www.modelscope.cn/models/<model_id>.git refs/heads/master

The three groups (monolithic / modular one-shot / modular collaborative)
share this same set of frozen weights; only the pipeline differs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR_NAME = "models"
PINNED_AT = "2026-08-20"
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 256
# Fixed smoke image shipped with the repository (perceiver models need one).
SMOKE_IMAGE_REL = "project/resources/image/test.jpg"
SMOKE_TEXT_PROMPT = "用一句话介绍职业规划。"

FROZEN_MODELS: List[Dict[str, str]] = [
    {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "5d854aab08710c16b980ec6d603d863b3821b915",
        "role": "monolithic",
        "kind": "vl",
        "pinned_at": PINNED_AT,
    },
    {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "ae9985b208c074c10cfbe3a61b5cb7268cdc9c53",
        "role": "perceiver",
        "kind": "vl",
        "pinned_at": PINNED_AT,
    },
    {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "2de2439ea21be1dc5cb21f22f88af07e43393cbb",
        "role": "reasoner",
        "kind": "text",
        "pinned_at": PINNED_AT,
    },
]

ROLE_TO_GROUPS = {
    "monolithic": "A: monolithic multimodal",
    "perceiver": "B: modular one-shot, C: modular collaborative",
    "reasoner": "B: modular one-shot, C: modular collaborative",
}


class ModelNotDownloadedError(FileNotFoundError):
    """Local weights are missing; download them on the L20 host first."""


def get_entry(model_id: str) -> Dict[str, str]:
    for entry in FROZEN_MODELS:
        if entry["model_id"] == model_id:
            return entry
    raise KeyError(f"unknown frozen model: {model_id}")


def local_model_dir(model_id: str, models_dir: Optional[Path] = None) -> Path:
    """Directory the pinned weights live in, given the ignored models root."""
    root = Path(models_dir or REPO_ROOT / MODELS_DIR_NAME)
    return root / model_id.split("/")[1]


class LocalModelAdapter:
    """Deterministic local-model backend for the primary experiment groups.

    Imports `torch`/`transformers` lazily inside :meth:`load` so that the
    offline runner and Windows core profile never need a GPU stack.

    Generation is greedy and seeded: the primary comparison must be
    reproducible, so no sampling knobs are exposed here.
    """

    def __init__(
        self,
        model_id: str,
        models_dir: Optional[Path] = None,
        seed: int = DEFAULT_SEED,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: Optional[str] = None,
    ):
        self.entry = get_entry(model_id)
        self.models_dir = Path(models_dir or REPO_ROOT / MODELS_DIR_NAME)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.device = device  # None = auto (cuda if available)
        self._model = None
        self._processor = None
        self._loaded_revision = None

    # -- lazy loading -----------------------------------------------------

    def load(self) -> "LocalModelAdapter":
        """Load the pinned local weights; never downloads anything."""
        target = local_model_dir(self.entry["model_id"], self.models_dir)
        if not (target / "config.json").exists():
            raise ModelNotDownloadedError(
                f"weights for {self.entry['model_id']} not found under {target}; "
                "run scripts/models/download_research_models.py on the L20 host"
            )
        import torch  # lazy GPU import
        import transformers

        transformers.set_seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        kwargs: Dict[str, Any] = {"device_map": device, "trust_remote_code": True}
        self._processor = transformers.AutoProcessor.from_pretrained(
            str(target), trust_remote_code=True
        )
        if self.entry["kind"] == "vl":
            self._model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
                str(target), **kwargs
            )
        else:
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                str(target), **kwargs
            )
        self._loaded_revision = self.entry["revision"]
        self._device = device
        return self

    # -- deterministic generation ------------------------------------------

    def generate(self, text_prompt: str, image_path: Optional[Path] = None) -> Dict[str, Any]:
        """One greedy, seeded completion; returns content plus telemetry."""
        if self._model is None:
            raise RuntimeError("call load() before generate()")

        import torch  # lazy GPU import

        processor = self._processor
        if self.entry["kind"] == "vl":
            image_path = image_path or (REPO_ROOT / SMOKE_IMAGE_REL)
            content: List[Dict[str, Any]] = []
            if image_path.exists():
                content.append(
                    {"type": "image", "image": str(image_path.resolve())}
                )
            content.append({"type": "text", "text": text_prompt})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[str(image_path)], return_tensors="pt"
            ).to(self._device)
        else:
            messages = [{"role": "user", "content": text_prompt}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor([text], return_tensors="pt").to(self._device)

        started = time.perf_counter()
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        elapsed_s = round(time.perf_counter() - started, 3)
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        output_text = processor.decode(output_ids, skip_special_tokens=True)

        peak_vram_mb = None
        if self._device == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2**20)

        return {
            "model_id": self.entry["model_id"],
            "revision": self._loaded_revision,
            "output_text": output_text,
            "elapsed_s": elapsed_s,
            "device": self._device,
            "peak_vram_mb": peak_vram_mb,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
        }

    # -- smoke entry --------------------------------------------------------

    @classmethod
    def run_smoke(
        cls, model_id: str, models_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Load one frozen model and run a single deterministic completion."""
        adapter = cls(model_id, models_dir=models_dir)
        adapter.load()
        return adapter.generate(SMOKE_TEXT_PROMPT)


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministic smoke inference for the frozen research models."
    )
    parser.add_argument(
        "--model",
        choices=["all"] + [m["model_id"] for m in FROZEN_MODELS],
        default="all",
        help="model to smoke (default: all three)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="models root (default: ignored ./models)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="JSON result dir (default: ignored ./data/experiments/smoke)",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir) if args.models_dir else None
    entries = (
        FROZEN_MODELS if args.model == "all" else [get_entry(args.model)]
    )
    results = []
    for entry in entries:
        result = LocalModelAdapter.run_smoke(entry["model_id"], models_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO_ROOT / "data" / "experiments" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    out_file = out_dir / f"smoke-{stamp}.json"
    out_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"results written to {out_file.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
