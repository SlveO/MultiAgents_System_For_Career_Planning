from __future__ import annotations

import uvicorn

from project.core.settings import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "project.api.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
