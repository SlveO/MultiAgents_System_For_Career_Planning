from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    brain_default_model: str = "deepseek-chat"
    brain_timeout_seconds: float = 45.0
    brain_retry_times: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings() -> AppSettings:
    return AppSettings()
