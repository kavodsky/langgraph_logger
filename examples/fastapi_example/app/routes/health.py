# app/routes/health.py
"""Health check endpoints."""

import time
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from app.core.database import DatabaseManager

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str = "1.0.0"
    database: Dict[str, Any]
    uptime_seconds: float


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic health check endpoint.

    Returns:
        Health status information
    """
    start_time = getattr(request.app.state, 'start_time', time.time())
    uptime = time.time() - start_time

    # Get database manager
    db_manager: DatabaseManager = request.app.state.db_manager

    # Check database health
    db_healthy = await db_manager.health_check()

    database_info = {
        "status": "healthy" if db_healthy else "unhealthy",
        "url_type": "sqlite" if "sqlite" in db_manager.database_url else "other"
    }

    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        timestamp=time.time(),
        database=database_info,
        uptime_seconds=uptime
    )


@router.get("/ready", response_model=Dict[str, str])
async def readiness_check(request: Request) -> Dict[str, str]:
    """Kubernetes readiness probe endpoint.

    Returns:
        Ready status

    Raises:
        HTTPException: If service is not ready
    """
    try:
        # Check if all required services are available
        db_manager: DatabaseManager = request.app.state.db_manager

        if not db_manager.repository:
            raise HTTPException(status_code=503, detail="Database not initialized")

        # Quick database check
        db_healthy = await db_manager.health_check()
        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database unhealthy")

        return {"status": "ready"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live", response_model=Dict[str, str])
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint.

    Returns:
        Alive status
    """
    return {"status": "alive"}