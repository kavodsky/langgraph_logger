# app/core/config.py
"""Application configuration."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Database settings
    database_url: str = Field(
        default="sqlite:///./langgraph_logger.db",
        env="DATABASE_URL"
    )

    # LangGraph Logger settings
    enable_console_logging: bool = Field(default=True, env="ENABLE_CONSOLE_LOGGING")
    enable_rich_output: bool = Field(default=True, env="ENABLE_RICH_OUTPUT")
    auto_save_state: bool = Field(default=True, env="AUTO_SAVE_STATE")
    recovery_checkpoint_interval: int = Field(default=5, env="RECOVERY_CHECKPOINT_INTERVAL")

    # Example graph settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()