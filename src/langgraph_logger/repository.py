# src/langgraph_logger/repository.py

"""Repository layer for database operations."""

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Generator

from sqlalchemy import create_engine, desc, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from .dto import (
    ExecutionStatus, NodeStatus, GraphExecutionCreate, GraphExecutionUpdate,
    NodeExecutionCreate, NodeExecutionUpdate, ExecutionStateCreate,
    ExecutionMetrics, RecoveryInfo
)
from .models import Base, GraphExecution, NodeExecution, ExecutionState
from .settings import GraphLoggerSettings

logger = logging.getLogger(__name__)


class GraphLoggerRepository:
    """Repository class for managing graph execution data persistence.

    This class provides a high-level interface for all database operations
    related to graph executions, node executions, and state management.
    """

    def __init__(self, settings: GraphLoggerSettings):
        """Initialize repository with database settings.

        Args:
            settings: Configuration settings for the logger
        """
        self.settings = settings
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            engine_kwargs = self.settings.get_database_engine_kwargs()
            self._engine = create_engine(self.settings.database_url, **engine_kwargs)
            logger.info(f"Created database engine for: {self.settings.database_url}")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping database tables: {e}")
            raise

    # Graph Execution Methods

    def create_graph_execution(self, data: GraphExecutionCreate) -> str:
        """Create a new graph execution record.

        Args:
            data: Graph execution creation data

        Returns:
            The ID of the created execution record
        """
        with self.get_session() as session:
            execution = GraphExecution(
                graph_name=data.graph_name,
                initial_state=data.initial_state,
                extra_metadata=data.extra_metadata,
                tags=data.tags,
                status=ExecutionStatus.RUNNING.value
            )

            session.add(execution)
            session.flush()  # Get the ID without committing

            logger.debug(f"Created graph execution: {execution.id}")
            return execution.id

    def update_graph_execution(self, execution_id: str, data: GraphExecutionUpdate) -> bool:
        """Update an existing graph execution record.

        Args:
            execution_id: ID of the execution to update
            data: Update data

        Returns:
            True if the record was updated, False if not found
        """
        with self.get_session() as session:
            execution = session.query(GraphExecution).filter(
                GraphExecution.id == execution_id
            ).first()

            if not execution:
                logger.warning(f"Graph execution not found: {execution_id}")
                return False

            # Update fields if provided
            if data.status is not None:
                execution.status = data.status.value
                if data.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                    execution.completed_at = datetime.utcnow()
                    if execution.started_at:
                        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

            if data.final_state is not None:
                execution.final_state = data.final_state

            if data.error_message is not None:
                execution.error_message = data.error_message

            if data.error_traceback is not None:
                execution.error_traceback = data.error_traceback

            if data.extra_metadata is not None:
                execution.extra_metadata = data.extra_metadata

            logger.debug(f"Updated graph execution: {execution_id}")
            return True

    def get_graph_execution(self, execution_id: str) -> Optional[GraphExecution]:
        """Get a graph execution by ID.

        Args:
            execution_id: ID of the execution to retrieve

        Returns:
            GraphExecution object or None if not found
        """
        with self.get_session() as session:
            return session.query(GraphExecution).filter(
                GraphExecution.id == execution_id
            ).first()

    def list_graph_executions(
            self,
            graph_name: Optional[str] = None,
            status: Optional[ExecutionStatus] = None,
            limit: int = 100,
            offset: int = 0,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> List[GraphExecution]:
        """List graph executions with optional filtering.

        Args:
            graph_name: Filter by graph name
            status: Filter by execution status
            limit: Maximum number of results
            offset: Number of results to skip
            start_date: Filter executions started after this date
            end_date: Filter executions started before this date

        Returns:
            List of GraphExecution objects
        """
        with self.get_session() as session:
            query = session.query(GraphExecution)

            # Apply filters
            if graph_name:
                query = query.filter(GraphExecution.graph_name == graph_name)

            if status:
                query = query.filter(GraphExecution.status == status.value)

            if start_date:
                query = query.filter(GraphExecution.started_at >= start_date)

            if end_date:
                query = query.filter(GraphExecution.started_at <= end_date)

            # Order by most recent first
            query = query.order_by(desc(GraphExecution.started_at))

            return query.offset(offset).limit(limit).all()

    # Node Execution Methods

    def create_node_execution(self, data: NodeExecutionCreate) -> str:
        """Create a new node execution record.

        Args:
            data: Node execution creation data

        Returns:
            The ID of the created node execution record
        """
        with self.get_session() as session:
            node_execution = NodeExecution(
                execution_id=data.execution_id,
                node_name=data.node_name,
                run_id=data.run_id,
                input_data=data.input_data,
                extra_metadata=data.extra_metadata,
                parent_run_id=data.parent_run_id,
                status=NodeStatus.RUNNING.value
            )

            session.add(node_execution)
            session.flush()

            logger.debug(f"Created node execution: {node_execution.id}")
            return node_execution.id

    def update_node_execution(self, run_id: str, data: NodeExecutionUpdate) -> bool:
        """Update an existing node execution record by run_id.

        Args:
            run_id: Run ID of the node execution to update
            data: Update data

        Returns:
            True if the record was updated, False if not found
        """
        with self.get_session() as session:
            node_execution = session.query(NodeExecution).filter(
                NodeExecution.run_id == run_id
            ).first()

            if not node_execution:
                logger.warning(f"Node execution not found: {run_id}")
                return False

            # Update fields if provided
            if data.status is not None:
                node_execution.status = data.status.value
                if data.status in (NodeStatus.COMPLETED, NodeStatus.FAILED):
                    node_execution.completed_at = datetime.utcnow()
                    if node_execution.started_at:
                        node_execution.duration_seconds = (
                                node_execution.completed_at - node_execution.started_at
                        ).total_seconds()

            if data.output_data is not None:
                node_execution.output_data = data.output_data

            if data.error_message is not None:
                node_execution.error_message = data.error_message

            if data.error_type is not None:
                node_execution.error_type = data.error_type

            if data.extra_metadata is not None:
                node_execution.extra_metadata = data.extra_metadata

            logger.debug(f"Updated node execution: {run_id}")
            return True

    def get_node_executions_for_graph(self, execution_id: str) -> List[NodeExecution]:
        """Get all node executions for a graph execution.

        Args:
            execution_id: ID of the graph execution

        Returns:
            List of NodeExecution objects
        """
        with self.get_session() as session:
            return session.query(NodeExecution).filter(
                NodeExecution.execution_id == execution_id
            ).order_by(NodeExecution.started_at).all()

    def get_active_node_executions(self, execution_id: str) -> List[NodeExecution]:
        """Get currently running node executions for a graph.

        Args:
            execution_id: ID of the graph execution

        Returns:
            List of currently running NodeExecution objects
        """
        with self.get_session() as session:
            return session.query(NodeExecution).filter(
                and_(
                    NodeExecution.execution_id == execution_id,
                    NodeExecution.status == NodeStatus.RUNNING.value
                )
            ).all()

    # Execution State Methods

    def create_execution_state(self, data: ExecutionStateCreate) -> str:
        """Create a new execution state snapshot.

        Args:
            data: Execution state creation data

        Returns:
            The ID of the created state record
        """
        with self.get_session() as session:
            # Get current sequence number
            max_seq = session.query(func.max(ExecutionState.sequence_number)).filter(
                ExecutionState.execution_id == data.execution_id
            ).scalar() or 0

            state = ExecutionState(
                execution_id=data.execution_id,
                checkpoint_name=data.checkpoint_name,
                sequence_number=max_seq + 1,
                state_data=data.state_data,
                extra_metadata=data.extra_metadata
            )

            session.add(state)
            session.flush()

            logger.debug(f"Created execution state: {state.id}")
            return state.id

    def get_latest_execution_state(self, execution_id: str) -> Optional[ExecutionState]:
        """Get the latest execution state for a graph execution.

        Args:
            execution_id: ID of the graph execution

        Returns:
            Latest ExecutionState object or None if not found
        """
        with self.get_session() as session:
            return session.query(ExecutionState).filter(
                ExecutionState.execution_id == execution_id
            ).order_by(desc(ExecutionState.sequence_number)).first()

    def get_recovery_states(self, execution_id: str) -> List[ExecutionState]:
        """Get all recovery states for a graph execution.

        Args:
            execution_id: ID of the graph execution

        Returns:
            List of recovery ExecutionState objects
        """
        with self.get_session() as session:
            return session.query(ExecutionState).filter(
                and_(
                    ExecutionState.execution_id == execution_id,
                    ExecutionState.is_recovery_point == True
                )
            ).order_by(ExecutionState.sequence_number).all()

    # Statistics and Analytics Methods

    def get_execution_metrics(self, execution_id: str) -> Optional[ExecutionMetrics]:
        """Calculate comprehensive metrics for a graph execution.

        Args:
            execution_id: ID of the graph execution

        Returns:
            ExecutionMetrics object or None if execution not found
        """
        with self.get_session() as session:
            execution = session.query(GraphExecution).filter(
                GraphExecution.id == execution_id
            ).first()

            if not execution:
                return None

            # Get node execution statistics
            nodes = session.query(NodeExecution).filter(
                NodeExecution.execution_id == execution_id
            ).all()

            total_duration = execution.duration_seconds or 0.0
            total_nodes = len(nodes)
            completed_nodes = sum(1 for n in nodes if n.is_completed)
            failed_nodes = sum(1 for n in nodes if n.is_failed)

            # Calculate node timings
            node_timings = {}
            durations = [n.duration_seconds for n in nodes if n.duration_seconds is not None]

            for node in nodes:
                if node.duration_seconds is not None:
                    if node.node_name not in node_timings:
                        node_timings[node.node_name] = {
                            'total_time': 0.0,
                            'executions': 0,
                            'avg_time': 0.0,
                            'min_time': float('inf'),
                            'max_time': 0.0
                        }

                    stats = node_timings[node.node_name]
                    stats['total_time'] += node.duration_seconds
                    stats['executions'] += 1
                    stats['min_time'] = min(stats['min_time'], node.duration_seconds)
                    stats['max_time'] = max(stats['max_time'], node.duration_seconds)
                    stats['avg_time'] = stats['total_time'] / stats['executions']

            # Find longest and shortest nodes
            longest_node = None
            longest_duration = 0.0
            shortest_node = None
            shortest_duration = float('inf')

            for node_name, stats in node_timings.items():
                if stats['max_time'] > longest_duration:
                    longest_duration = stats['max_time']
                    longest_node = node_name
                if stats['min_time'] < shortest_duration:
                    shortest_duration = stats['min_time']
                    shortest_node = node_name

            # Error summary
            error_summary = []
            for node in nodes:
                if node.is_failed and node.error_message:
                    error_summary.append({
                        'node_name': node.node_name,
                        'error_type': node.error_type,
                        'error_message': node.error_message,
                        'run_id': node.run_id
                    })

            return ExecutionMetrics(
                execution_id=execution_id,
                total_duration=total_duration,
                total_nodes=total_nodes,
                completed_nodes=completed_nodes,
                failed_nodes=failed_nodes,
                success_rate=(completed_nodes / max(1, total_nodes)) * 100,
                average_node_duration=sum(durations) / max(1, len(durations)),
                longest_node=longest_node,
                longest_node_duration=longest_duration if longest_node else None,
                shortest_node=shortest_node,
                shortest_node_duration=shortest_duration if shortest_node and shortest_duration != float(
                    'inf') else None,
                parallel_execution_peak=execution.max_parallel_nodes,
                error_summary=error_summary,
                node_timings=node_timings
            )

    def get_recovery_info(self, execution_id: str) -> Optional[RecoveryInfo]:
        """Get recovery information for a failed execution.

        Args:
            execution_id: ID of the graph execution

        Returns:
            RecoveryInfo object or None if execution not found
        """
        with self.get_session() as session:
            execution = session.query(GraphExecution).filter(
                GraphExecution.id == execution_id
            ).first()

            if not execution:
                return None

            # Get recovery states
            recovery_states = self.get_recovery_states(execution_id)
            latest_state = self.get_latest_execution_state(execution_id)

            # Get failed nodes
            failed_nodes = session.query(NodeExecution).filter(
                and_(
                    NodeExecution.execution_id == execution_id,
                    NodeExecution.status == NodeStatus.FAILED.value
                )
            ).all()

            failed_node_names = [node.node_name for node in failed_nodes]

            # Determine if recovery is possible
            can_recover = len(recovery_states) > 0 and execution.is_failed

            # Find last successful checkpoint
            last_checkpoint = None
            recovery_state = None
            if recovery_states:
                last_recovery = recovery_states[-1]
                last_checkpoint = last_recovery.checkpoint_name
                recovery_state = last_recovery.state_data

            available_checkpoints = [state.checkpoint_name for state in recovery_states]

            return RecoveryInfo(
                execution_id=execution_id,
                last_successful_checkpoint=last_checkpoint or "none",
                available_checkpoints=available_checkpoints,
                recovery_state=recovery_state or {},
                failed_nodes=failed_node_names,
                can_recover=can_recover,
                recovery_instructions=self._generate_recovery_instructions(execution, failed_nodes, recovery_states)
            )

    def _generate_recovery_instructions(
            self,
            execution: GraphExecution,
            failed_nodes: List[NodeExecution],
            recovery_states: List[ExecutionState]
    ) -> str:
        """Generate human-readable recovery instructions."""
        if not recovery_states:
            return "No recovery states available. Execution must be restarted from the beginning."

        last_state = recovery_states[-1]
        instructions = [
            f"Execution can be recovered from checkpoint '{last_state.checkpoint_name}'.",
            f"Failed nodes: {[n.node_name for n in failed_nodes]}",
            f"Recovery will resume from state saved at {last_state.created_at}."
        ]

        return " ".join(instructions)

    def cleanup_old_executions(self, days: int) -> int:
        """Clean up old execution records.

        Args:
            days: Number of days to keep records

        Returns:
            Number of deleted records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            # Count records to be deleted
            count = session.query(GraphExecution).filter(
                GraphExecution.created_at < cutoff_date
            ).count()

            # Delete old executions (cascades to related records)
            session.query(GraphExecution).filter(
                GraphExecution.created_at < cutoff_date
            ).delete()

            logger.info(f"Cleaned up {count} old execution records")
            return count

    def update_execution_stats(self, execution_id: str) -> None:
        """Update cached statistics for a graph execution.

        Args:
            execution_id: ID of the graph execution to update
        """
        with self.get_session() as session:
            execution = session.query(GraphExecution).filter(
                GraphExecution.id == execution_id
            ).first()

            if not execution:
                return

            # Count node statistics
            nodes = session.query(NodeExecution).filter(
                NodeExecution.execution_id == execution_id
            ).all()

            execution.total_nodes = len(nodes)
            execution.completed_nodes = sum(1 for n in nodes if n.is_completed)
            execution.failed_nodes = sum(1 for n in nodes if n.is_failed)

            # Calculate max parallel nodes (approximate)
            # This is a simplified calculation - in practice you might want to track this more precisely
            running_at_same_time = session.query(NodeExecution).filter(
                NodeExecution.execution_id == execution_id,
                NodeExecution.status == NodeStatus.RUNNING.value
            ).count()

            execution.max_parallel_nodes = max(execution.max_parallel_nodes, running_at_same_time)

            logger.debug(f"Updated stats for execution: {execution_id}")

    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")