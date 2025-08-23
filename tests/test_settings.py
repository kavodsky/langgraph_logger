"""Test settings configuration."""

import pytest
from pydantic import ValidationError

from langgraph_logger.settings import GraphLoggerSettings


def test_default_settings():
    """Test default settings creation."""
    settings = GraphLoggerSettings.create_default()

    assert settings.database_url == "sqlite:///./langgraph_executions.db"
    assert settings.log_level == "INFO"
    assert settings.enable_rich_output is True
    assert settings.auto_save_state is True


def test_custom_settings():
    """Test custom settings validation."""
    settings = GraphLoggerSettings(
        database_url="postgresql://localhost/test",
        log_level="DEBUG",
        enable_rich_output=False,
        recovery_checkpoint_interval=10
    )

    assert settings.database_url == "postgresql://localhost/test"
    assert settings.log_level == "DEBUG"
    assert settings.enable_rich_output is False
    assert settings.recovery_checkpoint_interval == 10


def test_invalid_log_level():
    """Test invalid log level validation."""
    with pytest.raises(ValidationError):
        GraphLoggerSettings(log_level="INVALID")


def test_database_type_detection():
    """Test database type detection methods."""
    sqlite_settings = GraphLoggerSettings(database_url="sqlite:///test.db")
    postgres_settings = GraphLoggerSettings(database_url="postgresql://localhost/test")
    mysql_settings = GraphLoggerSettings(database_url="mysql://localhost/test")

    assert sqlite_settings.is_sqlite is True
    assert sqlite_settings.is_postgres is False
    assert sqlite_settings.is_mysql is False

    assert postgres_settings.is_postgres is True
    assert postgres_settings.is_sqlite is False

    assert mysql_settings.is_mysql is True
    assert mysql_settings.is_sqlite is False