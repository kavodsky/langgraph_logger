# src/langgraph_logger/utils.py

"""Utility functions for LangGraph Logger."""

import json
import logging
import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from .callback import GraphExecutionCallback
from .repository import GraphLoggerRepository
from .settings import GraphLoggerSettings

logger = logging.getLogger(__name__)


def create_logger_callback(
        graph_name: str,
        database_url: Optional[str] = None,
        enable_rich_output: bool = True,
        auto_save_state: bool = True,
        **kwargs
) -> GraphExecutionCallback:
    """Create a configured GraphExecutionCallback with sensible defaults.

    Args:
        graph_name: Name of the graph being executed
        database_url: Database URL override
        enable_rich_output: Enable rich console output
        auto_save_state: Enable automatic state saving
        **kwargs: Additional arguments for GraphExecutionCallback

    Returns:
        Configured GraphExecutionCallback instance
    """
    settings = GraphLoggerSettings.create_default()

    if database_url:
        settings.database_url = database_url

    settings.enable_rich_output = enable_rich_output
    settings.auto_save_state = auto_save_state

    return GraphExecutionCallback(
        graph_name=graph_name,
        settings=settings,
        **kwargs
    )


@contextmanager
def logged_graph_execution(
        graph_name: str,
        initial_state: Dict[str, Any],
        database_url: Optional[str] = None,
        tags: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **callback_kwargs
) -> Generator[GraphExecutionCallback, None, None]:
    """Context manager for logged graph execution.

    Usage:
        with logged_graph_execution("my_graph", initial_state) as callback:
            result = graph.invoke(initial_state, config={"callbacks": [callback]})

    Args:
        graph_name: Name of the graph being executed
        initial_state: Initial state of the graph
        database_url: Database URL override
        tags: Tags for the execution
        metadata: Additional metadata
        **callback_kwargs: Additional callback arguments

    Yields:
        GraphExecutionCallback instance
    """
    callback = create_logger_callback(
        graph_name=graph_name,
        database_url=database_url,
        initial_state=initial_state,
        tags=tags,
        extra_metadata=metadata,  # Updated field name
        **callback_kwargs
    )

    try:
        yield callback
    finally:
        # Context manager exit will handle finalization
        pass


def setup_logging(
        level: str = "INFO",
        format_string: Optional[str] = None,
        enable_rich: bool = True
) -> None:
    """Setup logging configuration for LangGraph Logger.

    Args:
        level: Logging level
        format_string: Custom format string
        enable_rich: Enable rich logging handler
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=sys.stdout
    )

    if enable_rich:
        try:
            from rich.logging import RichHandler

            # Remove default handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # Add rich handler
            rich_handler = RichHandler(rich_tracebacks=True)
            rich_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.root.addHandler(rich_handler)

        except ImportError:
            logger.warning("Rich not available, using standard logging")


def validate_database_connection(database_url: str) -> bool:
    """Validate that database connection works.

    Args:
        database_url: Database URL to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        settings = GraphLoggerSettings.create_default()
        settings.database_url = database_url

        repository = GraphLoggerRepository(settings)

        # Test connection by creating tables
        repository.create_tables()
        repository.close()

        return True

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def export_execution_data(
        execution_id: str,
        output_file: str,
        database_url: Optional[str] = None,
        include_states: bool = True,
        include_nodes: bool = True
) -> bool:
    """Export execution data to JSON file.

    Args:
        execution_id: ID of execution to export
        output_file: Path to output JSON file
        database_url: Database URL override
        include_states: Include saved states in export
        include_nodes: Include node execution details

    Returns:
        True if export successful, False otherwise
    """
    try:
        settings = GraphLoggerSettings.create_default()
        if database_url:
            settings.database_url = database_url

        repository = GraphLoggerRepository(settings)

        # Get execution data
        execution = repository.get_graph_execution(execution_id)
        if not execution:
            logger.error(f"Execution not found: {execution_id}")
            return False

        export_data = {
            "execution": {
                "id": execution.id,
                "graph_name": execution.graph_name,
                "status": execution.status,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "initial_state": execution.initial_state,
                "final_state": execution.final_state,
                "error_message": execution.error_message,
                "total_nodes": execution.total_nodes,
                "completed_nodes": execution.completed_nodes,
                "failed_nodes": execution.failed_nodes,
                "max_parallel_nodes": execution.max_parallel_nodes,
                "extra_metadata": execution.extra_metadata,  # Updated field name
                "tags": execution.tags
            }
        }

        # Add node executions
        if include_nodes:
            nodes = repository.get_node_executions_for_graph(execution_id)
            export_data["nodes"] = [
                {
                    "id": node.id,
                    "node_name": node.node_name,
                    "run_id": node.run_id,
                    "status": node.status,
                    "started_at": node.started_at.isoformat(),
                    "completed_at": node.completed_at.isoformat() if node.completed_at else None,
                    "duration_seconds": node.duration_seconds,
                    "input_data": node.input_data,
                    "output_data": node.output_data,
                    "error_message": node.error_message,
                    "error_type": node.error_type,
                    "extra_metadata": node.extra_metadata  # Updated field name
                }
                for node in nodes
            ]

        # Add states
        if include_states:
            states = repository.get_recovery_states(execution_id)
            export_data["states"] = [
                {
                    "id": state.id,
                    "checkpoint_name": state.checkpoint_name,
                    "sequence_number": state.sequence_number,
                    "created_at": state.created_at.isoformat(),
                    "state_data": state.state_data,
                    "extra_metadata": state.extra_metadata  # Updated field name
                }
                for state in states
            ]

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Execution data exported to: {output_file}")
        repository.close()
        return True

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False


def import_execution_data(
        input_file: str,
        database_url: Optional[str] = None,
        overwrite: bool = False
) -> Optional[str]:
    """Import execution data from JSON file.

    Args:
        input_file: Path to input JSON file
        database_url: Database URL override
        overwrite: Overwrite existing execution if it exists

    Returns:
        Execution ID if import successful, None otherwise
    """
    try:
        settings = GraphLoggerSettings.create_default()
        if database_url:
            settings.database_url = database_url

        repository = GraphLoggerRepository(settings)

        # Read file
        with open(input_file, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        execution_data = import_data.get("execution")
        if not execution_data:
            logger.error("Invalid import file: missing execution data")
            return None

        # Check if execution exists
        existing = repository.get_graph_execution(execution_data["id"])
        if existing and not overwrite:
            logger.error(f"Execution {execution_data['id']} already exists. Use --overwrite to replace.")
            return None

        # TODO: Implement full import functionality
        # This would require creating new records with the imported data
        # For now, this is a placeholder for the structure

        logger.warning("Import functionality not fully implemented yet")
        repository.close()
        return None

    except Exception as e:
        logger.error(f"Import failed: {e}")
        return None


def get_execution_summary(
        execution_id: str,
        database_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get a comprehensive summary of an execution.

    Args:
        execution_id: ID of execution to summarize
        database_url: Database URL override

    Returns:
        Summary dictionary or None if execution not found
    """
    try:
        settings = GraphLoggerSettings.create_default()
        if database_url:
            settings.database_url = database_url

        repository = GraphLoggerRepository(settings)

        execution = repository.get_graph_execution(execution_id)
        if not execution:
            return None

        metrics = repository.get_execution_metrics(execution_id)
        recovery_info = repository.get_recovery_info(execution_id)

        summary = {
            "execution_id": execution.id,
            "graph_name": execution.graph_name,
            "status": execution.status,
            "duration": execution.duration_seconds,
            "success_rate": execution.success_rate,
            "total_nodes": execution.total_nodes,
            "parallel_peak": execution.max_parallel_nodes,
            "has_errors": execution.failed_nodes > 0,
            "can_recover": recovery_info.can_recover if recovery_info else False,
            "metrics": metrics.model_dump() if metrics else None,
            "recovery": recovery_info.model_dump() if recovery_info else None
        }

        repository.close()
        return summary

    except Exception as e:
        logger.error(f"Error getting execution summary: {e}")
        return None


class ExecutionRecover:
    """Helper class for recovering failed executions."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize recovery helper.

        Args:
            database_url: Database URL override
        """
        self.settings = GraphLoggerSettings.create_default()
        if database_url:
            self.settings.database_url = database_url

        self.repository = GraphLoggerRepository(self.settings)

    def can_recover(self, execution_id: str) -> bool:
        """Check if execution can be recovered.

        Args:
            execution_id: ID of execution to check

        Returns:
            True if recovery is possible
        """
        recovery_info = self.repository.get_recovery_info(execution_id)
        return recovery_info.can_recover if recovery_info else False

    def get_recovery_state(self, execution_id: str, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get state for recovery.

        Args:
            execution_id: ID of execution to recover
            checkpoint_name: Specific checkpoint to recover from (uses latest if None)

        Returns:
            Recovery state dictionary or None if not available
        """
        if checkpoint_name:
            # Get specific checkpoint
            states = self.repository.get_recovery_states(execution_id)
            for state in states:
                if state.checkpoint_name == checkpoint_name:
                    return state.state_data
            return None
        else:
            # Get latest state
            state = self.repository.get_latest_execution_state(execution_id)
            return state.state_data if state else None

    def list_checkpoints(self, execution_id: str) -> list:
        """List available checkpoints for recovery.

        Args:
            execution_id: ID of execution

        Returns:
            List of checkpoint names
        """
        states = self.repository.get_recovery_states(execution_id)
        return [state.checkpoint_name for state in states]

    def close(self) -> None:
        """Close database connection."""
        self.repository.close()