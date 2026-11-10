"""Explicit retention cleanup; the live API also cleans every minute."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from project.web.storage import WebStore


def main():
    load_dotenv()
    root = Path(os.environ.get("CAREER_WEB_DATA_DIR", Path(__file__).resolve().parents[2] / "data/web"))
    store = WebStore(os.environ.get("CAREER_WEB_DB", root / "web.sqlite3"), os.environ.get("CAREER_WEB_UPLOADS", root / "uploads"))
    store.initialize()
    print(json.dumps(store.cleanup(), ensure_ascii=False))


if __name__ == "__main__":
    main()
