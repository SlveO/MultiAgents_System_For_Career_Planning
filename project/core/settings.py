from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    brain_default_model: str = "deepseek-v4-flash"
    brain_timeout_seconds: float = 45.0
    brain_retry_times: int = 2
    local_model_device: str = "cuda:0"
    vision_model_path: str = "./models/Qwen3-VL-2B-Instruct"
    guidance_model_path: str = ""
    feedback_model_path: str = ""
    guidance_max_rounds: int = Field(default=4, ge=0, le=16)
    audio_model_path: str = "./models/whisper-small"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", case_sensitive=False, extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
