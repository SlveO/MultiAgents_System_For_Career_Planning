from __future__ import annotations

import os
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_JWT_SECRET = "change-me-in-production"


class AppSettings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    brain_default_model: str = "deepseek-v4-flash"
    brain_timeout_seconds: float = 45.0
    brain_retry_times: int = 2

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    jwt_secret_key: str = _DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    if settings.jwt_secret_key == _DEFAULT_JWT_SECRET and "JWT_SECRET_KEY" not in os.environ:
        generated = os.urandom(32).hex()
        settings.jwt_secret_key = generated
        print(
            "[settings] WARNING: JWT_SECRET_KEY is using the default insecure value. "
            f"Auto-generated a random key for this session: {generated[:8]}...",
            file=sys.stderr,
        )
        print(
            "[settings] Set JWT_SECRET_KEY environment variable for production use.",
            file=sys.stderr,
        )
    return settings
