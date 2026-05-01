from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    brain_default_model: str = "deepseek-chat"
    brain_timeout_seconds: float = 45.0
    brain_retry_times: int = 2

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings() -> AppSettings:
    return AppSettings()
