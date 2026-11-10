"""Run only on loopback; TLS ingress is supplied by the trial tunnel."""
from __future__ import annotations

import os

from dotenv import load_dotenv


def main():
    load_dotenv()
    # Never silently use GPU0 or enable implicit model/API retries on the Web.
    os.environ.setdefault("LOCAL_MODEL_DEVICE", "cuda:1")
    if os.environ["LOCAL_MODEL_DEVICE"] != "cuda:1":
        raise RuntimeError("Web 服务按当前方案只使用 GPU1；请显式配置 LOCAL_MODEL_DEVICE=cuda:1。")
    os.environ["BRAIN_RETRY_TIMES"] = "0"
    import uvicorn
    from .app import create_app
    port = int(os.environ.get("CAREER_WEB_PORT", "8787"))
    if not 1024 <= port <= 65535:
        raise RuntimeError("CAREER_WEB_PORT 必须在1024至65535之间。")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, workers=1, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
