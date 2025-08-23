"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path

from langgraph_logger.settings import GraphLoggerSettings
from langgraph_logger.repository import GraphLoggerRepository


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_settings(temp_db):
    """Create test settings with temporary database."""
    return GraphLoggerSettings(
        database_url=temp_db,
        enable_rich_output=False,  # Disable for testing
        enable_console_logging=False,  # Disable for testing
        auto_save_state=True,
        log_level="DEBUG"
    )


@pytest.fixture
def test_repository(test_settings):
    """Create test repository with initialized database."""
    repository = GraphLoggerRepository(test_settings)
    repository.create_tables()
    yield repository
    repository.close()