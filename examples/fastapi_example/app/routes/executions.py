# app/routes/executions.py
"""Execution management and monitoring endpoints."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)
router = APIRouter()


class ExecutionListResponse(BaseModel):
    """Response model for execution list."""
    executions: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int


class ExecutionDetailResponse(BaseModel):
    """Response model for execution details."""
    execution: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None


class CleanupRequest(BaseModel):
    """Request model for cleanup operations."""
    days: int = Field(default=30, ge=1, le=365, description="Number of days to keep")
    dry_run: bool = Field(default=True, description="Whether to perform a dry run")


@router.get("/", response_model=ExecutionListResponse)
async def list_executions(
        app_request: Request,
        graph_name: Optional[str] = Query(None, description="Filter by graph name"),
        limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        page: int = Query(1, ge=1, description="Page number (alternative to offset)")
) -> ExecutionListResponse:
    """List graph executions with pagination and filtering.

    Args:
        app_request: FastAPI request object
        graph_name: Optional filter by graph name
        limit: Maximum number of results
        offset: Offset for pagination
        page: Page number (alternative to offset)

    Returns:
        Paginated list of executions
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        # Calculate offset from page if provided
        if page > 1:
            offset = (page - 1) * limit

        executions = await graph_service.list_executions(
            graph_name=graph_name,
            limit=limit,
            offset=offset
        )

        return ExecutionListResponse(
            executions=executions,
            total_count=len(executions),  # This is approximate since we don't have total count
            page=page,
            page_size=limit
        )

    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve executions")


@router.get("/{execution_id}", response_model=ExecutionDetailResponse)
async def get_execution_details(
        execution_id: str,
        app_request: Request,
        include_metrics: bool = Query(True, description="Whether to include detailed metrics")
) -> ExecutionDetailResponse:
    """Get detailed information about a specific execution.

    Args:
        execution_id: ID of the execution
        app_request: FastAPI request object
        include_metrics: Whether to include detailed metrics

    Returns:
        Detailed execution information

    Raises:
        HTTPException: If execution not found
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        execution_data = await graph_service.get_execution_status(execution_id)

        if not execution_data:
            raise HTTPException(status_code=404, detail="Execution not found")

        response_data = {
            "execution": execution_data.get("execution", {}),
        }

        if include_metrics and "metrics" in execution_data:
            response_data["metrics"] = execution_data["metrics"]

        return ExecutionDetailResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution details")


@router.get("/{execution_id}/nodes")
async def get_execution_nodes(
        execution_id: str,
        app_request: Request
) -> Dict[str, Any]:
    """Get node execution details for a specific graph execution.

    Args:
        execution_id: ID of the execution
        app_request: FastAPI request object

    Returns:
        Node execution information

    Raises:
        HTTPException: If execution not found
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service
        repository = graph_service.repository

        # Get node executions from repository
        import asyncio
        nodes = await asyncio.get_event_loop().run_in_executor(
            None,
            repository.get_node_executions_for_graph,
            execution_id
        )

        if not nodes:
            # Check if execution exists
            execution_data = await graph_service.get_execution_status(execution_id)
            if not execution_data:
                raise HTTPException(status_code=404, detail="Execution not found")

        # Convert to dict format
        nodes_data = [
            node.dict() if hasattr(node, 'dict') else node.__dict__
            for node in nodes
        ]

        return {
            "execution_id": execution_id,
            "nodes": nodes_data,
            "total_nodes": len(nodes_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution nodes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve node executions")


@router.get("/{execution_id}/states")
async def get_execution_states(
        execution_id: str,
        app_request: Request,
        recovery_only: bool = Query(False, description="Only return recovery states")
) -> Dict[str, Any]:
    """Get state snapshots for a specific execution.

    Args:
        execution_id: ID of the execution
        app_request: FastAPI request object
        recovery_only: Whether to return only recovery states

    Returns:
        State snapshots information

    Raises:
        HTTPException: If execution not found
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service
        repository = graph_service.repository

        import asyncio

        if recovery_only:
            states = await asyncio.get_event_loop().run_in_executor(
                None,
                repository.get_recovery_states,
                execution_id
            )
        else:
            # Get latest state (we could extend this to get all states)
            latest_state = await asyncio.get_event_loop().run_in_executor(
                None,
                repository.get_latest_execution_state,
                execution_id
            )
            states = [latest_state] if latest_state else []

        states_data = [
            state.dict() if hasattr(state, 'dict') else state.__dict__
            for state in states
        ]

        return {
            "execution_id": execution_id,
            "states": states_data,
            "total_states": len(states_data),
            "recovery_only": recovery_only
        }

    except Exception as e:
        logger.error(f"Error getting execution states: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution states")


@router.get("/stats/summary")
async def get_execution_statistics(
        app_request: Request,
        days: int = Query(7, ge=1, le=90, description="Number of days to include in stats")
) -> Dict[str, Any]:
    """Get execution statistics summary.

    Args:
        app_request: FastAPI request object
        days: Number of days to include in statistics

    Returns:
        Statistics summary
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get recent executions
        executions = await graph_service.list_executions(limit=1000)

        # Filter by date range (this is simplified - in production you'd do this at DB level)
        recent_executions = []
        for execution in executions:
            exec_date = execution.get('started_at')
            if exec_date and isinstance(exec_date, str):
                try:
                    exec_datetime = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                    if start_date <= exec_datetime <= end_date:
                        recent_executions.append(execution)
                except ValueError:
                    continue
            elif exec_date and isinstance(exec_date, datetime):
                if start_date <= exec_date <= end_date:
                    recent_executions.append(execution)

        # Calculate statistics
        total_executions = len(recent_executions)
        completed_executions = len([e for e in recent_executions if e.get('status') == 'completed'])
        failed_executions = len([e for e in recent_executions if e.get('status') == 'failed'])
        running_executions = len([e for e in recent_executions if e.get('status') == 'running'])

        # Calculate success rate
        success_rate = (completed_executions / max(1, total_executions)) * 100

        # Group by graph name
        graph_stats = {}
        for execution in recent_executions:
            graph_name = execution.get('graph_name', 'unknown')
            if graph_name not in graph_stats:
                graph_stats[graph_name] = {'total': 0, 'completed': 0, 'failed': 0}

            graph_stats[graph_name]['total'] += 1
            if execution.get('status') == 'completed':
                graph_stats[graph_name]['completed'] += 1
            elif execution.get('status') == 'failed':
                graph_stats[graph_name]['failed'] += 1

        return {
            "period": {
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "totals": {
                "total_executions": total_executions,
                "completed_executions": completed_executions,
                "failed_executions": failed_executions,
                "running_executions": running_executions,
                "success_rate": round(success_rate, 2)
            },
            "by_graph": graph_stats
        }

    except Exception as e:
        logger.error(f"Error getting execution statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_old_executions(
        request: CleanupRequest,
        app_request: Request
) -> Dict[str, Any]:
    """Clean up old execution records.

    Args:
        request: Cleanup request parameters
        app_request: FastAPI request object

    Returns:
        Cleanup operation results
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service
        repository = graph_service.repository

        if request.dry_run:
            # For dry run, just count what would be deleted
            import asyncio
            from datetime import datetime, timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=request.days)

            # This is a simplified version - you'd want a proper count query
            all_executions = await graph_service.list_executions(limit=10000)

            old_count = 0
            for execution in all_executions:
                exec_date = execution.get('started_at')
                if exec_date and isinstance(exec_date, str):
                    try:
                        exec_datetime = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                        if exec_datetime < cutoff_date:
                            old_count += 1
                    except ValueError:
                        continue
                elif exec_date and isinstance(exec_date, datetime):
                    if exec_date < cutoff_date:
                        old_count += 1

            return {
                "dry_run": True,
                "days": request.days,
                "would_delete": old_count,
                "cutoff_date": cutoff_date.isoformat()
            }
        else:
            # Perform actual cleanup
            import asyncio
            deleted_count = await asyncio.get_event_loop().run_in_executor(
                None,
                repository.cleanup_old_executions,
                request.days
            )

            return {
                "dry_run": False,
                "days": request.days,
                "deleted_count": deleted_count
            }

    except Exception as e:
        logger.error(f"Error in cleanup operation: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")


@router.delete("/{execution_id}")
async def delete_execution(
        execution_id: str,
        app_request: Request
) -> Dict[str, str]:
    """Delete a specific execution and all its related data.

    Args:
        execution_id: ID of the execution to delete
        app_request: FastAPI request object

    Returns:
        Deletion confirmation

    Raises:
        HTTPException: If execution not found or deletion fails
    """
    try:
        graph_service: GraphService = app_request.app.state.graph_service

        # Check if execution exists
        execution_data = await graph_service.get_execution_status(execution_id)
        if not execution_data:
            raise HTTPException(status_code=404, detail="Execution not found")

        # Note: This would require implementing a delete method in the repository
        # For now, we'll return a not implemented error
        raise HTTPException(
            status_code=501,
            detail="Execution deletion not implemented. Use cleanup endpoint for bulk operations."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete execution")