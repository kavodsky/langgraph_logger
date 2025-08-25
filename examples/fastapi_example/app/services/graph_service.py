# app/services/graph_service.py
"""Service layer for graph execution management."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langgraph_logger.core.callback import GraphExecutionCallback
from langgraph_logger.database.repository import GraphLoggerRepository
from langgraph_logger.utils.helpers import logged_graph_execution

from app.graph.example_graph import create_example_graph, GraphState

logger = logging.getLogger(__name__)


class GraphExecutionError(Exception):
    """Custom exception for graph execution errors."""
    pass


class GraphService:
    """Service for managing graph executions with logging."""

    def __init__(self, repository: GraphLoggerRepository):
        """Initialize graph service.

        Args:
            repository: Database repository instance
        """
        self.repository = repository
        self._graph = None

    @property
    def graph(self):
        """Get or create the compiled graph."""
        if self._graph is None:
            self._graph = create_example_graph()
        return self._graph

    async def execute_graph(
            self,
            initial_state: Dict[str, Any],
            graph_name: str = "example_graph",
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute graph with logging.

        Args:
            initial_state: Initial state for graph execution
            graph_name: Name of the graph being executed
            tags: Optional tags for the execution
            metadata: Optional metadata for the execution
            execution_id: Optional custom execution ID

        Returns:
            Dictionary containing execution results and metadata

        Raises:
            GraphExecutionError: If graph execution fails
        """
        execution_id = execution_id or str(uuid4())
        tags = tags or []
        metadata = metadata or {}

        logger.info(f"Starting graph execution: {execution_id}")

        try:
            # Create graph state
            state = GraphState(**initial_state)

            # Execute in thread pool to avoid blocking FastAPI
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._execute_with_logging,
                state,
                graph_name,
                tags,
                metadata,
                execution_id
            )

            logger.info(f"Graph execution completed: {execution_id}")
            return result

        except Exception as e:
            logger.error(f"Graph execution failed: {execution_id}, error: {e}")
            raise GraphExecutionError(f"Graph execution failed: {str(e)}") from e

    def _execute_with_logging(
            self,
            state: GraphState,
            graph_name: str,
            tags: List[str],
            metadata: Dict[str, Any],
            execution_id: str
    ) -> Dict[str, Any]:
        """Execute graph with logging (synchronous)."""

        # Create callback with repository settings
        callback = GraphExecutionCallback(
            graph_name=graph_name,
            settings=self.repository.settings,
            repository=self.repository,
            initial_state=state.dict(),
            tags=tags,
            extra_metadata={
                **metadata,
                "custom_execution_id": execution_id,
                "service_version": "1.0.0"
            }
        )

        try:
            # Execute graph with callback
            with callback:
                final_state = self.graph.invoke(
                    state,
                    config={"callbacks": [callback]}
                )

            # Get execution summary
            summary = callback.get_execution_summary()

            return {
                "execution_id": callback.execution_id,
                "custom_execution_id": execution_id,
                "final_state": final_state.dict() if hasattr(final_state, 'dict') else final_state,
                "summary": summary,
                "status": "completed"
            }

        except Exception as e:
            # Callback context manager will handle finalization with error
            return {
                "execution_id": callback.execution_id,
                "custom_execution_id": execution_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed"
            }

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status and details.

        Args:
            execution_id: ID of execution to check

        Returns:
            Execution details or None if not found
        """
        try:
            execution = await asyncio.get_event_loop().run_in_executor(
                None,
                self.repository.get_graph_execution,
                execution_id
            )

            if not execution:
                return None

            # Get additional metrics
            metrics = await asyncio.get_event_loop().run_in_executor(
                None,
                self.repository.get_execution_metrics,
                execution_id
            )

            return {
                "execution": execution.dict() if hasattr(execution, 'dict') else execution.__dict__,
                "metrics": metrics.dict() if metrics and hasattr(metrics, 'dict') else None
            }

        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return None

    async def list_executions(
            self,
            graph_name: Optional[str] = None,
            limit: int = 50,
            offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List recent executions.

        Args:
            graph_name: Optional filter by graph name
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of execution summaries
        """
        try:
            executions = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.repository.list_graph_executions(
                    graph_name=graph_name,
                    limit=limit,
                    offset=offset
                )
            )

            return [
                execution.dict() if hasattr(execution, 'dict') else execution.__dict__
                for execution in executions
            ]

        except Exception as e:
            logger.error(f"Error listing executions: {e}")
            return []

    async def get_execution_recovery_info(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get recovery information for a failed execution.

        Args:
            execution_id: ID of execution to check

        Returns:
            Recovery information or None if not available
        """
        try:
            recovery_info = await asyncio.get_event_loop().run_in_executor(
                None,
                self.repository.get_recovery_info,
                execution_id
            )

            if not recovery_info:
                return None

            return recovery_info.dict() if hasattr(recovery_info, 'dict') else recovery_info.__dict__

        except Exception as e:
            logger.error(f"Error getting recovery info: {e}")
            return None

    async def create_manual_checkpoint(
            self,
            execution_id: str,
            checkpoint_name: str,
            state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a manual checkpoint during execution.

        Args:
            execution_id: ID of execution
            checkpoint_name: Name for the checkpoint
            state: Optional state data

        Returns:
            True if checkpoint was created successfully
        """
        try:
            # This would typically be called during an active execution
            # For this example, we'll create a state record directly
            from langgraph_logger.dto.state import ExecutionStateCreate

            checkpoint_data = ExecutionStateCreate(
                execution_id=execution_id,
                checkpoint_name=checkpoint_name,
                state_data=state or {},
                extra_metadata={"manual": True, "created_via_api": True}
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.repository.create_execution_state,
                checkpoint_data
            )

            return bool(result)

        except Exception as e:
            logger.error(f"Error creating manual checkpoint: {e}")
            return False