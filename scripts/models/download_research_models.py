"""Deterministic download of the three frozen research models (L20 Ubuntu only).

Weights go to the ignored local `models/` directory with the exact pinned
ModelScope revisions from `project.experiments.local_models.FROZEN_MODELS`.
Running this script on Windows is refused by design: research weights and
primary runs are restricted to the networked L20 Ubuntu host.

After each download a manifest `models/research_models_manifest.json` is
written with the pinned revision and resolved size so the run record stays
reproducible without consulting any remote API.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from project.experiments.local_models import FROZEN_MODELS, local_model_dir  # noqa: E402

ALLOWED_PLATFORM = "Linux"


class PlatformGuardError(RuntimeError):
    """Raised when a download is attempted outside the approved L20 host."""


def ensure_allowed_platform(system: Optional[str] = None) -> None:
    """Refuse to run anywhere but the approved host; `system` is injectable
    so tests can exercise the guard without touching platform internals."""
    system = system or platform.system()
    if system != ALLOWED_PLATFORM:
        raise PlatformGuardError(
            f"research weights may only be downloaded on the networked L20 Ubuntu "
            f"host (current platform: {system}). See docs/experiments.md and "
            f"docs/l20-setup.md."
        )


def download_entry(entry: Dict[str, str], models_dir: Path) -> Path:
    """Download one pinned model via ModelScope (lazy GPU-profile import)."""
    from modelscope import snapshot_download  # lazy: requirements-gpu.txt

    target = local_model_dir(entry["model_id"], models_dir)
    resolved = snapshot_download(
        entry["model_id"],
        revision=entry["revision"],
        local_dir=str(target),
    )
    return Path(resolved)


def write_manifest(entries: List[Dict[str, str]], models_dir: Path) -> Path:
    manifest = models_dir / "research_models_manifest.json"
    records = []
    for entry in entries:
        target = local_model_dir(entry["model_id"], models_dir)
        if not (target / "config.json").exists():
            raise RuntimeError(
                f"download failed or incomplete: {target} has no config.json"
            )
        size_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        records.append(
            {
                "model_id": entry["model_id"],
                "revision": entry["revision"],
                "role": entry["role"],
                "kind": entry["kind"],
                "local_dir": f"models/{entry['model_id'].split('/')[1]}",
                "size_bytes": size_bytes,
                "downloaded_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            }
        )
    manifest.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the frozen research models on the L20 Ubuntu host."
    )
    parser.add_argument(
        "--model",
        choices=["all"] + [m["model_id"] for m in FROZEN_MODELS],
        default="all",
        help="model to download (default: all three)",
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "models"),
        help="models root (default: ignored ./models)",
    )
    args = parser.parse_args(argv)

    # Refuse early, before any network or GPU dependency is touched.
    ensure_allowed_platform()

    entries = (
        FROZEN_MODELS if args.model == "all" else [e for e in FROZEN_MODELS if e["model_id"] == args.model]
    )
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        print(
            f"downloading {entry['model_id']} @ {entry['revision']} "
            f"(role={entry['role']}) ..."
        )
        target = download_entry(entry, models_dir)
        print(f"  -> {target}")

    manifest = write_manifest(entries, models_dir)
    print(f"manifest written to {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlatformGuardError as exc:
        print(f"[blocked] {exc}", file=sys.stderr)
        raise SystemExit(2)
