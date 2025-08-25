# main.py
"""FastAPI application demonstrating LangGraph Logger usage."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import DatabaseManager
from app.routes import executions, graphs, health
from app.services.graph_service import GraphService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan events handler."""
    settings = get_settings()

    # Startup
    print("Starting FastAPI application...")

    # Initialize database
    db_manager = DatabaseManager(settings.database_url)
    await db_manager.initialize()
    app.state.db_manager = db_manager

    # Initialize graph service
    graph_service = GraphService(db_manager.get_repository())
    app.state.graph_service = graph_service

    print("Application started successfully")

    yield

    # Shutdown
    print("Shutting down application...")
    await db_manager.close()
    print("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="LangGraph Logger Demo",
        description="FastAPI application demonstrating LangGraph Logger usage",
        version="1.0.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(graphs.router, prefix="/api/v1/graphs", tags=["graphs"])
    app.include_router(executions.router, prefix="/api/v1/executions", tags=["executions"])

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )