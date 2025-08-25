# app/routes/graphs.py
"""Graph execution endpoints."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from app.services.graph_service import GraphService, GraphExecutionError

logger = logging.getLogger(__name__)
router = APIRouter()


class GraphExecutionRequest(BaseModel):
    """Request model for graph execution."""
    initial_state: Dict[str, Any] = Field(..., description="Initial state for graph execution")
    graph_name: str = Field(default="example_graph", description="Name of the graph to execute")
    tags: Optional[List[str]] = Field(default=None, description="Tags for the execution")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    execution_id: Optional[str] = Field(default=None, description="Custom execution ID")


class GraphExecutionResponse(BaseModel):
    """Response model for graph execution."""
    execution_id: str
    custom_execution_id: Optional[str] = None
    status: str
    final_state: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class CheckpointRequest(BaseModel):
    """Request model for manual checkpoint creation."""
    checkpoint_name: str = Field(..., description="Name for the checkpoint")
    state: Optional[Dict[str, Any]] = Field(default=None, description="State data for checkpoint")


@router.post("/execute", response_model=GraphExecutionResponse)
async def execute_graph(
        request: GraphExecutionRequest,
        background_tasks: BackgroundTasks,
        app_request: Request
) -> GraphExecutionResponse:
    """Execute a graph with the provided initial state.

    Args:
        request: Graph execution request
        background_tasks: FastAPI background tasks
        app_request: FastAPI request object

    Returns:
        Execution response with results

    Raises:
        HTTPException: If execution fails
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        result = await graph_service.execute_graph(
            initial_state=request.initial_state,
            graph_name=request.graph_name,
            tags=request.tags,
            metadata=request.metadata,
            execution_id=request.execution_id
        )

        return GraphExecutionResponse(**result)

    except GraphExecutionError as e:
        logger.error(f"Graph execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in graph execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/execute-async", response_model=Dict[str, str])
async def execute_graph_async(
        request: GraphExecutionRequest,
        background_tasks: BackgroundTasks,
        app_request: Request
) -> Dict[str, str]:
    """Execute a graph asynchronously in the background.

    Args:
        request: Graph execution request
        background_tasks: FastAPI background tasks
        app_request: FastAPI request object

    Returns:
        Task information
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())

        # Add background task
        background_tasks.add_task(
            _execute_graph_background,
            graph_service,
            request,
            task_id
        )

        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Graph execution started in background"
        }

    except Exception as e:
        logger.error(f"Error starting background execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to start background execution")


async def _execute_graph_background(
        graph_service: GraphService,
        request: GraphExecutionRequest,
        task_id: str
) -> None:
    """Background task for graph execution."""
    try:
        logger.info(f"Starting background graph execution: {task_id}")

        result = await graph_service.execute_graph(
            initial_state=request.initial_state,
            graph_name=request.graph_name,
            tags=request.tags or [] + [f"background_task:{task_id}"],
            metadata={**(request.metadata or {}), "background_task_id": task_id},
            execution_id=request.execution_id
        )

        logger.info(f"Background graph execution completed: {task_id}")

    except Exception as e:
        logger.error(f"Background graph execution failed: {task_id}, error: {e}")


@router.get("/templates")
async def get_graph_templates() -> Dict[str, Any]:
    """Get available graph templates and example initial states.

    Returns:
        Dictionary of available templates
    """
    return {
        "templates": {
            "example_graph": {
                "description": "Demonstration graph with multiple processing paths",
                "example_initial_state": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Process this example input",
                            "timestamp": 1234567890
                        }
                    ],
                    "current_step": "start",
                    "processed_count": 0,
                    "result": {},
                    "metadata": {"source": "api_template"}
                },
                "nodes": [
                    "input_processor",
                    "analysis",
                    "decision",
                    "enhancement",
                    "validation",
                    "error_handling",
                    "finalization"
                ],
                "paths": {
                    "high_confidence": "input_processor -> analysis -> decision -> enhancement -> finalization",
                    "medium_confidence": "input_processor -> analysis -> decision -> validation -> finalization",
                    "low_confidence": "input_processor -> analysis -> decision -> error_handling -> finalization"
                }
            }
        }
    }


@router.post("/{execution_id}/checkpoint", response_model=Dict[str, Any])
async def create_manual_checkpoint(
        execution_id: str,
        request: CheckpointRequest,
        app_request: Request
) -> Dict[str, Any]:
    """Create a manual checkpoint for an execution.

    Args:
        execution_id: ID of the execution
        request: Checkpoint request
        app_request: FastAPI request object

    Returns:
        Checkpoint creation result

    Raises:
        HTTPException: If checkpoint creation fails
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        success = await graph_service.create_manual_checkpoint(
            execution_id=execution_id,
            checkpoint_name=request.checkpoint_name,
            state=request.state
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to create checkpoint")

        return {
            "success": True,
            "checkpoint_name": request.checkpoint_name,
            "execution_id": execution_id,
            "message": "Checkpoint created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating checkpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{execution_id}/recovery", response_model=Dict[str, Any])
async def get_recovery_info(
        execution_id: str,
        app_request: Request
) -> Dict[str, Any]:
    """Get recovery information for a failed execution.

    Args:
        execution_id: ID of the execution
        app_request: FastAPI request object

    Returns:
        Recovery information

    Raises:
        HTTPException: If execution not found
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        recovery_info = await graph_service.get_execution_recovery_info(execution_id)

        if not recovery_info:
            raise HTTPException(status_code=404, detail="Execution not found or no recovery info available")

        return recovery_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recovery info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def graph_service_health(app_request: Request) -> Dict[str, Any]:
    """Check graph service health.

    Args:
        app_request: FastAPI request object

    Returns:
        Health status information
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        # Test basic graph creation
        graph = graph_service.graph

        return {
            "status": "healthy",
            "graph_available": graph is not None,
            "repository_connected": graph_service.repository is not None
        }

    except Exception as e:
        logger.error(f"Graph service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }