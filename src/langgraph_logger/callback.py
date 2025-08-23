# src/langgraph_logger/callback.py

"""Main callback handler for LangGraph execution logging."""

import logging
import time
import traceback
from collections import defaultdict
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.table import Table

from .dto import (
    ExecutionStatus, NodeStatus, GraphExecutionCreate, GraphExecutionUpdate,
    NodeExecutionCreate, NodeExecutionUpdate, ExecutionStateCreate
)
from .repository import GraphLoggerRepository
from .settings import GraphLoggerSettings

logger = logging.getLogger(__name__)


class GraphExecutionCallback(BaseCallbackHandler):
    """Advanced callback handler for LangGraph execution tracking with database persistence.

    This callback handler provides comprehensive logging, monitoring, and state management
    for LangGraph workflows. It supports parallel execution tracking, automatic state
    snapshots for recovery, and detailed metrics collection.

    Features:
    - Database persistence of execution logs and states
    - Parallel node execution tracking
    - Automatic recovery state snapshots
    - Rich console output (optional)
    - Comprehensive metrics and statistics
    - Configurable cleanup and maintenance
    """

    def __init__(
            self,
            graph_name: str,
            settings: Optional[GraphLoggerSettings] = None,
            repository: Optional[GraphLoggerRepository] = None,
            initial_state: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the callback handler.

        Args:
            graph_name: Name of the graph being executed
            settings: Configuration settings (uses defaults if not provided)
            repository: Database repository (created if not provided)
            initial_state: Initial state of the graph execution
            tags: Tags for categorizing the execution
            extra_metadata: Additional metadata for the execution
        """
        self.graph_name = graph_name
        self.settings = settings or GraphLoggerSettings.create_default()
        self.repository = repository or GraphLoggerRepository(self.settings)

        # Rich console for output (if enabled)
        self.console = Console() if self.settings.enable_rich_output else None

        # Thread safety
        self._lock = Lock()

        # Execution tracking
        self.execution_id: Optional[str] = None
        self.graph_start_time: Optional[float] = None
        self.initial_state = initial_state or {}
        self.tags = tags
        self.extra_metadata = extra_metadata or {}

        # Node execution tracking
        self.node_start_times: Dict[str, float] = {}  # run_id -> start_time
        self.node_names: Dict[str, str] = {}  # run_id -> node_name
        self.active_runs: Set[str] = set()  # Currently running run_ids
        self.completed_runs: Set[str] = set()  # Completed run_ids
        self.failed_runs: Set[str] = set()  # Failed run_ids

        # State management
        self.current_state: Dict[str, Any] = self.initial_state.copy()
        self.checkpoint_counter = 0
        self.last_checkpoint_node_count = 0

        # Statistics
        self.max_concurrent_nodes = 0
        self.total_nodes_started = 0
        self.total_nodes_completed = 0
        self.total_nodes_failed = 0

        # Initialize database tables if needed
        try:
            self.repository.create_tables()
        except Exception as e:
            logger.warning(f"Could not create database tables: {e}")

    def _ensure_execution_started(self) -> None:
        """Ensure graph execution is recorded in database."""
        if self.execution_id is None:
            with self._lock:
                if self.execution_id is None:  # Double-check locking
                    try:
                        execution_data = GraphExecutionCreate(
                            graph_name=self.graph_name,
                            initial_state=self.initial_state,
                            extra_metadata=self.extra_metadata,
                            tags=self.tags
                        )
                        self.execution_id = self.repository.create_graph_execution(execution_data)
                        self.graph_start_time = time.perf_counter()

                        if self.settings.enable_console_logging:
                            self._log_graph_start()

                    except Exception as e:
                        logger.error(f"Failed to create graph execution record: {e}")
                        # Continue without database logging
                        self.execution_id = f"local_{int(time.time())}"
                        self.graph_start_time = time.perf_counter()

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a node/chain starts execution."""
        self._ensure_execution_started()

        run_id = kwargs.get("run_id", f"unknown_run_{time.time()}")
        node_name = self._extract_node_name(serialized, kwargs)

        with self._lock:
            # Track this node execution
            self.active_runs.add(run_id)
            self.node_start_times[run_id] = time.perf_counter()
            self.node_names[run_id] = node_name
            self.total_nodes_started += 1

            # Update concurrent execution tracking
            current_concurrent = len(self.active_runs)
            if current_concurrent > self.max_concurrent_nodes:
                self.max_concurrent_nodes = current_concurrent

        # Log to database
        if self.execution_id and not self.execution_id.startswith("local_"):
            try:
                node_data = NodeExecutionCreate(
                    execution_id=self.execution_id,
                    node_name=node_name,
                    run_id=run_id,
                    input_data=inputs if len(str(inputs)) < 10000 else {"_truncated": True},
                    extra_metadata=kwargs.get("extra_metadata")
                )
                self.repository.create_node_execution(node_data)
            except Exception as e:
                logger.error(f"Failed to create node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_start(node_name, inputs, run_id)

        # Update current state with inputs (simple merge)
        if inputs and self.settings.auto_save_state:
            self.current_state.update(inputs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a node/chain ends successfully."""
        run_id = kwargs.get("run_id", "unknown_run")

        with self._lock:
            if run_id not in self.node_start_times or run_id not in self.node_names:
                logger.warning(f"Node end called for unknown run_id: {run_id}")
                return

            # Calculate execution time
            end_time = time.perf_counter()
            execution_time = end_time - self.node_start_times[run_id]
            node_name = self.node_names[run_id]

            # Update tracking
            self.active_runs.discard(run_id)
            self.completed_runs.add(run_id)
            self.total_nodes_completed += 1

            # Clean up
            del self.node_start_times[run_id]
            del self.node_names[run_id]

        # Update database
        if self.execution_id and not self.execution_id.startswith("local_"):
            try:
                update_data = NodeExecutionUpdate(
                    status=NodeStatus.COMPLETED,
                    output_data=outputs if len(str(outputs)) < 10000 else {"_truncated": True}
                )
                self.repository.update_node_execution(run_id, update_data)
                self.repository.update_execution_stats(self.execution_id)
            except Exception as e:
                logger.error(f"Failed to update node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_end(node_name, outputs, execution_time, run_id)

        # Update current state with outputs
        if outputs and self.settings.auto_save_state:
            self.current_state.update(outputs)

        # Create checkpoint if needed
        self._maybe_create_checkpoint()

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a node/chain encounters an error."""
        run_id = kwargs.get("run_id", "unknown_run")

        with self._lock:
            if run_id not in self.node_start_times or run_id not in self.node_names:
                logger.warning(f"Node error called for unknown run_id: {run_id}")
                return

            # Calculate time until error
            error_time = time.perf_counter()
            execution_time = error_time - self.node_start_times[run_id]
            node_name = self.node_names[run_id]

            # Update tracking
            self.active_runs.discard(run_id)
            self.failed_runs.add(run_id)
            self.total_nodes_failed += 1

            # Clean up
            del self.node_start_times[run_id]
            del self.node_names[run_id]

        # Update database
        if self.execution_id and not self.execution_id.startswith("local_"):
            try:
                update_data = NodeExecutionUpdate(
                    status=NodeStatus.FAILED,
                    error_message=str(error),
                    error_type=type(error).__name__
                )
                self.repository.update_node_execution(run_id, update_data)
                self.repository.update_execution_stats(self.execution_id)
            except Exception as e:
                logger.error(f"Failed to update node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_error(node_name, error, execution_time, run_id)

        # Create error checkpoint
        self._create_error_checkpoint(node_name, error)

    def finalize_execution(self, final_state: Optional[Dict[str, Any]] = None,
                           error: Optional[Exception] = None) -> None:
        """Finalize the graph execution.

        Args:
            final_state: Final state of the graph execution
            error: Error that caused execution failure (if any)
        """
        if self.execution_id is None:
            return

        # Determine final status
        if error:
            status = ExecutionStatus.FAILED
            error_message = str(error)
            error_traceback = traceback.format_exc()
        elif self.total_nodes_failed > 0:
            status = ExecutionStatus.FAILED
            error_message = f"{self.total_nodes_failed} nodes failed"
            error_traceback = None
        else:
            status = ExecutionStatus.COMPLETED
            error_message = None
            error_traceback = None

        # Update final state
        if final_state:
            self.current_state.update(final_state)

        # Update database
        if not self.execution_id.startswith("local_"):
            try:
                update_data = GraphExecutionUpdate(
                    status=status,
                    final_state=self.current_state,
                    error_message=error_message,
                    error_traceback=error_traceback
                )
                self.repository.update_graph_execution(self.execution_id, update_data)

                # Create final state checkpoint
                if self.settings.auto_save_state:
                    self._create_checkpoint("final_state", self.current_state)

            except Exception as e:
                logger.error(f"Failed to finalize execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_graph_end(status, error)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution.

        Returns:
            Dictionary containing execution summary
        """
        total_time = time.perf_counter() - self.graph_start_time if self.graph_start_time else 0

        return {
            "execution_id": self.execution_id,
            "graph_name": self.graph_name,
            "total_time": total_time,
            "total_nodes_started": self.total_nodes_started,
            "total_nodes_completed": self.total_nodes_completed,
            "total_nodes_failed": self.total_nodes_failed,
            "success_rate": (self.total_nodes_completed / max(1, self.total_nodes_started)) * 100,
            "max_concurrent_nodes": self.max_concurrent_nodes,
            "active_runs": len(self.active_runs),
            "checkpoint_count": self.checkpoint_counter
        }

    def print_execution_summary(self) -> None:
        """Print a comprehensive summary of the graph execution."""
        if self.graph_start_time is None:
            if self.settings.enable_console_logging:
                print("No graph execution detected.")
            return

        summary = self.get_execution_summary()

        if self.settings.enable_rich_output and self.console:
            self._print_rich_summary(summary)
        else:
            self._print_plain_summary(summary)

    def create_manual_checkpoint(self, name: str, state: Optional[Dict[str, Any]] = None) -> None:
        """Manually create a checkpoint.

        Args:
            name: Name for the checkpoint
            state: State to save (uses current state if not provided)
        """
        checkpoint_state = state or self.current_state
        self._create_checkpoint(name, checkpoint_state, is_manual=True)

    def get_recovery_info(self) -> Optional[Dict[str, Any]]:
        """Get recovery information for the current execution.

        Returns:
            Recovery information dictionary or None if not available
        """
        if not self.execution_id or self.execution_id.startswith("local_"):
            return None

        try:
            recovery_info = self.repository.get_recovery_info(self.execution_id)
            return recovery_info.model_dump() if recovery_info else None
        except Exception as e:
            logger.error(f"Failed to get recovery info: {e}")
            return None

    def _extract_node_name(self, serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
        """Extract node name from callback arguments."""
        return (
                (serialized or {}).get("name") or
                kwargs.get("name") or
                kwargs.get("extra_metadata", {}).get("langgraph_node") or
                f"unknown_node_{int(time.time())}"
        )

    def _maybe_create_checkpoint(self) -> None:
        """Create a checkpoint if conditions are met."""
        if not self.settings.auto_save_state:
            return

        nodes_since_checkpoint = self.total_nodes_completed - self.last_checkpoint_node_count

        if nodes_since_checkpoint >= self.settings.recovery_checkpoint_interval:
            checkpoint_name = f"auto_checkpoint_{self.checkpoint_counter + 1}"
            self._create_checkpoint(checkpoint_name, self.current_state)
            self.last_checkpoint_node_count = self.total_nodes_completed

    def _create_checkpoint(self, name: str, state: Dict[str, Any], is_manual: bool = False) -> None:
        """Create a state checkpoint.

        Args:
            name: Name for the checkpoint
            state: State data to save
            is_manual: Whether this is a manually created checkpoint
        """
        if not self.execution_id or self.execution_id.startswith("local_"):
            return

        try:
            # Prepare metadata
            metadata = {
                "is_manual": is_manual,
                "nodes_completed": self.total_nodes_completed,
                "nodes_failed": self.total_nodes_failed,
                "max_concurrent": self.max_concurrent_nodes
            }

            checkpoint_data = ExecutionStateCreate(
                execution_id=self.execution_id,
                checkpoint_name=name,
                state_data=state,
                extra_metadata=metadata
            )

            self.repository.create_execution_state(checkpoint_data)
            self.checkpoint_counter += 1

            logger.debug(f"Created checkpoint: {name}")

        except Exception as e:
            logger.error(f"Failed to create checkpoint {name}: {e}")

    def _create_error_checkpoint(self, node_name: str, error: Exception) -> None:
        """Create a checkpoint when an error occurs."""
        if not self.settings.auto_save_state:
            return

        checkpoint_name = f"error_before_{node_name}_{int(time.time())}"
        error_metadata = {
            "error_node": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "is_error_checkpoint": True
        }

        # Add error info to current state
        state_with_error = self.current_state.copy()
        state_with_error["_error_info"] = error_metadata

        self._create_checkpoint(checkpoint_name, state_with_error)

    # Logging methods

    def _log_graph_start(self) -> None:
        """Log the start of graph execution."""
        if self.settings.enable_rich_output and self.console:
            self.console.print(f"🚀 [bold blue]Graph '{self.graph_name}' execution started[/bold blue]")
        else:
            logger.info(f"🚀 Graph '{self.graph_name}' execution started")

    def _log_node_start(self, node_name: str, inputs: Dict[str, Any], run_id: str) -> None:
        """Log when a node starts execution."""
        concurrent_info = f" [Concurrent: {len(self.active_runs)}]" if len(self.active_runs) > 1 else ""
        run_id_short = str(run_id)[:8] + "..." if len(str(run_id)) > 8 else str(run_id)

        if self.settings.enable_rich_output and self.console:
            self.console.print(
                f"▶️ [bold yellow]Node '{node_name}' started[/bold yellow] "
                f"(Run ID: {run_id_short}){concurrent_info}"
            )

            # Show input keys (not full content to avoid clutter)
            if inputs:
                input_info = f"Input keys: {list(inputs.keys())}"
                self.console.print(f"   📤 {input_info}")
        else:
            logger.info(f"▶️ Node '{node_name}' started (Run ID: {run_id_short}){concurrent_info}")

    def _log_node_end(self, node_name: str, outputs: Dict[str, Any], execution_time: float, run_id: str) -> None:
        """Log when a node completes successfully."""
        run_id_short = str(run_id)[:8] + "..." if len(str(run_id)) > 8 else str(run_id)

        if self.settings.enable_rich_output and self.console:
            self.console.print(
                f"✅ [bold green]Node '{node_name}' completed[/bold green] "
                f"in [bold cyan]{execution_time:.3f}s[/bold cyan] (Run ID: {run_id_short})"
            )

            if outputs:
                output_info = f"Output keys: {list(outputs.keys())}"
                self.console.print(f"   📥 {output_info}")
        else:
            logger.info(f"✅ Node '{node_name}' completed in {execution_time:.3f}s (Run ID: {run_id_short})")

    def _log_node_error(self, node_name: str, error: Exception, execution_time: float, run_id: str) -> None:
        """Log when a node encounters an error."""
        error_type = type(error).__name__
        run_id_short = str(run_id)[:8] + "..." if len(str(run_id)) > 8 else str(run_id)

        if self.settings.enable_rich_output and self.console:
            self.console.print(
                f"❌ [bold red]Node '{node_name}' failed[/bold red] "
                f"after [bold cyan]{execution_time:.3f}s[/bold cyan] (Run ID: {run_id_short})"
            )
            self.console.print(f"   🚨 Error: [red]{error_type}: {str(error)}[/red]")
        else:
            logger.error(f"❌ Node '{node_name}' failed after {execution_time:.3f}s (Run ID: {run_id_short})")
            logger.error(f"Error: {error_type}: {str(error)}")

    def _log_graph_end(self, status: ExecutionStatus, error: Optional[Exception] = None) -> None:
        """Log the end of graph execution."""
        if self.settings.enable_rich_output and self.console:
            if status == ExecutionStatus.COMPLETED:
                self.console.print(f"🎉 [bold green]Graph '{self.graph_name}' completed successfully[/bold green]")
            else:
                self.console.print(f"💥 [bold red]Graph '{self.graph_name}' failed[/bold red]")
                if error:
                    self.console.print(f"   🚨 Error: [red]{str(error)}[/red]")
        else:
            if status == ExecutionStatus.COMPLETED:
                logger.info(f"🎉 Graph '{self.graph_name}' completed successfully")
            else:
                logger.error(f"💥 Graph '{self.graph_name}' failed")
                if error:
                    logger.error(f"Error: {str(error)}")

    def _print_rich_summary(self, summary: Dict[str, Any]) -> None:
        """Print rich formatted execution summary."""
        if not self.console:
            return

        # Main summary table
        summary_table = Table(title=f"📊 Execution Summary: {self.graph_name}")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Execution ID", str(summary["execution_id"]))
        summary_table.add_row("Total Time", f"{summary['total_time']:.3f}s")
        summary_table.add_row("Nodes Started", str(summary["total_nodes_started"]))
        summary_table.add_row("Nodes Completed", str(summary["total_nodes_completed"]))
        summary_table.add_row("Nodes Failed", str(summary["total_nodes_failed"]))
        summary_table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        summary_table.add_row("Max Concurrent", str(summary["max_concurrent_nodes"]))
        summary_table.add_row("Checkpoints Created", str(summary["checkpoint_count"]))

        self.console.print(summary_table)

    def _print_plain_summary(self, summary: Dict[str, Any]) -> None:
        """Print plain text execution summary."""
        print(f"\n📊 Execution Summary: {self.graph_name}")
        print(f"   Execution ID: {summary['execution_id']}")
        print(f"   Total time: {summary['total_time']:.3f}s")
        print(f"   Nodes started: {summary['total_nodes_started']}")
        print(f"   Nodes completed: {summary['total_nodes_completed']}")
        print(f"   Nodes failed: {summary['total_nodes_failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Max concurrent: {summary['max_concurrent_nodes']}")
        print(f"   Checkpoints: {summary['checkpoint_count']}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finalize execution."""
        error = None
        if exc_type is not None:
            error = exc_val or Exception(f"Execution failed with {exc_type.__name__}")

        self.finalize_execution(error=error)

        if self.settings.enable_console_logging:
            self.print_execution_summary()

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'repository') and self.repository:
                self.repository.close()
        except Exception:
            pass  # Ignore cleanup errors