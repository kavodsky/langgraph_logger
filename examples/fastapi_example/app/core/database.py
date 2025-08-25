# app/core/database.py
"""Database management for the FastAPI application."""

import asyncio
import logging
from typing import Optional

from langgraph_logger.config.settings import GraphLoggerSettings
from langgraph_logger.database.repository import GraphLoggerRepository

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and repository instances."""

    def __init__(self, database_url: str):
        """Initialize database manager.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.settings: Optional[GraphLoggerSettings] = None
        self.repository: Optional[GraphLoggerRepository] = None

    async def initialize(self) -> None:
        """Initialize database and create tables."""
        try:
            # Create settings
            self.settings = GraphLoggerSettings.create_default()
            self.settings.database_url = self.database_url

            # Create repository and initialize tables
            self.repository = GraphLoggerRepository(self.settings)

            # Run database initialization in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self.repository.create_tables
            )

            logger.info(f"Database initialized successfully: {self.database_url}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_repository(self) -> GraphLoggerRepository:
        """Get repository instance.

        Returns:
            GraphLoggerRepository instance

        Raises:
            RuntimeError: If database not initialized
        """
        if self.repository is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.repository

    def get_settings(self) -> GraphLoggerSettings:
        """Get settings instance.

        Returns:
            GraphLoggerSettings instance

        Raises:
            RuntimeError: If database not initialized
        """
        if self.settings is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.settings

    async def close(self) -> None:
        """Close database connections."""
        if self.repository:
            await asyncio.get_event_loop().run_in_executor(
                None, self.repository.close
            )
            logger.info("Database connections closed")

    async def health_check(self) -> bool:
        """Check database health.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            if not self.repository:
                return False

            # Try to get execution count (simple query)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.repository.list_graph_executions(limit=1)
            )
            return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False