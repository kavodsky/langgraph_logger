# src/langgraph_logger/settings.py

"""Settings configuration for LangGraph Logger using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphLoggerSettings(BaseSettings):
    """Configuration settings for the LangGraph Logger.

    This class handles all configuration options for the logging system,
    including database connection, logging levels, and performance settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LANGGRAPH_LOGGER_",
        case_sensitive=False,
        extra="ignore"
    )

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./langgraph_executions.db",
        description="Database URL for storing execution logs"
    )

    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Database connection pool size"
    )

    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum database connection overflow"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    enable_rich_output: bool = Field(
        default=True,
        description="Enable rich console output for logging"
    )

    enable_console_logging: bool = Field(
        default=True,
        description="Enable console logging output"
    )

    # State Management
    auto_save_state: bool = Field(
        default=True,
        description="Automatically save graph state during execution"
    )

    state_compression: bool = Field(
        default=False,
        description="Enable compression for large state objects"
    )

    max_state_size_mb: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum state size in MB before warning"
    )

    # Performance Settings
    batch_insert_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of records to insert in a single batch"
    )

    cleanup_old_executions_days: Optional[int] = Field(
        default=30,
        ge=1,
        description="Number of days to keep old execution records (None to keep all)"
    )

    enable_metrics_collection: bool = Field(
        default=True,
        description="Enable detailed metrics collection"
    )

    # Recovery Settings
    enable_recovery: bool = Field(
        default=True,
        description="Enable graph execution recovery from saved states"
    )

    recovery_checkpoint_interval: int = Field(
        default=5,
        ge=1,
        description="Number of nodes between recovery checkpoints"
    )

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.upper()

    @validator("database_url")
    def validate_database_url(cls, v: str) -> str:
        """Validate and process database URL."""
        if v.startswith("sqlite:///"):
            # Create directory for SQLite database if it doesn't exist
            db_path = Path(v.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def is_sqlite(self) -> bool:
        """Check if the configured database is SQLite."""
        return self.database_url.startswith("sqlite:")

    @property
    def is_postgres(self) -> bool:
        """Check if the configured database is PostgreSQL."""
        return self.database_url.startswith(("postgres:", "postgresql:"))

    @property
    def is_mysql(self) -> bool:
        """Check if the configured database is MySQL."""
        return self.database_url.startswith("mysql:")

    def get_database_engine_kwargs(self) -> dict:
        """Get database engine configuration kwargs."""
        kwargs = {
            "echo": self.database_echo,
        }

        # Only add pool settings for non-SQLite databases
        if not self.is_sqlite:
            kwargs.update({
                "pool_size": self.database_pool_size,
                "max_overflow": self.database_max_overflow,
            })

        return kwargs

    @classmethod
    def create_default(cls) -> "GraphLoggerSettings":
        """Create default settings instance."""
        return cls()

    def __str__(self) -> str:
        """String representation of settings."""
        return (
            f"GraphLoggerSettings("
            f"database={self.database_url}, "
            f"log_level={self.log_level}, "
            f"rich_output={self.enable_rich_output})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()