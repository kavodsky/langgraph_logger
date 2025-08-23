# src/langgraph_logger/models.py

"""SQLAlchemy models for LangGraph Logger database schema."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    String, Text, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator, TEXT

from .dto import ExecutionStatus, NodeStatus

# Base class for all models
Base = declarative_base()


class JSONType(TypeDecorator):
    """Custom JSON type that works across different database backends."""

    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> Optional[str]:
        """Convert Python object to JSON string."""
        if value is None:
            return None
        return json.dumps(value, default=str, ensure_ascii=False)

    def process_result_value(self, value: str, dialect) -> Any:
        """Convert JSON string to Python object."""
        if value is None:
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class GraphExecution(Base):
    """Model representing a complete graph execution session.

    This table stores high-level information about graph executions,
    including status, timing, and state information.
    """

    __tablename__ = "graph_executions"

    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)

    # Graph identification
    graph_name = Column(String(255), nullable=False, index=True)

    # Execution status and timing
    status = Column(String(20), nullable=False, default=ExecutionStatus.RUNNING.value, index=True)
    started_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    duration_seconds = Column(Float, nullable=True)

    # State management
    initial_state = Column(JSONType, nullable=False)
    final_state = Column(JSONType, nullable=True)
    state_size_bytes = Column(Integer, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)

    # Statistics
    total_nodes = Column(Integer, default=0)
    completed_nodes = Column(Integer, default=0)
    failed_nodes = Column(Integer, default=0)
    max_parallel_nodes = Column(Integer, default=1)

    # Additional information (renamed from metadata to avoid conflicts)
    extra_metadata = Column(JSONType, nullable=True)
    tags = Column(JSONType, nullable=True)  # List of strings

    # Audit fields
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    node_executions = relationship("NodeExecution", back_populates="graph_execution", cascade="all, delete-orphan")
    execution_states = relationship("ExecutionState", back_populates="graph_execution", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('running', 'completed', 'failed', 'cancelled', 'paused')", name="valid_status"),
        CheckConstraint("completed_nodes >= 0", name="non_negative_completed_nodes"),
        CheckConstraint("failed_nodes >= 0", name="non_negative_failed_nodes"),
        CheckConstraint("total_nodes >= 0", name="non_negative_total_nodes"),
        Index("idx_graph_executions_status_started", "status", "started_at"),
        Index("idx_graph_executions_graph_name_started", "graph_name", "started_at"),
    )

    @validates('status')
    def validate_status(self, key: str, value: str) -> str:
        """Validate execution status."""
        valid_statuses = {status.value for status in ExecutionStatus}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status: {value}. Must be one of: {valid_statuses}")
        return value

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100.0

    @property
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status == ExecutionStatus.RUNNING.value

    @property
    def is_completed(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == ExecutionStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == ExecutionStatus.FAILED.value

    def __repr__(self) -> str:
        """String representation of GraphExecution."""
        return f"<GraphExecution(id={self.id}, graph_name={self.graph_name}, status={self.status})>"


class NodeExecution(Base):
    """Model representing individual node executions within a graph.

    This table stores detailed information about each node execution,
    supporting parallel execution tracking through run_id.
    """

    __tablename__ = "node_executions"

    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)

    # Foreign key to graph execution
    execution_id = Column(String(36), ForeignKey("graph_executions.id"), nullable=False, index=True)

    # Node identification
    node_name = Column(String(255), nullable=False, index=True)
    run_id = Column(String(255), nullable=False, unique=True, index=True)
    parent_run_id = Column(String(255), nullable=True, index=True)  # For nested node calls

    # Execution status and timing
    status = Column(String(20), nullable=False, default=NodeStatus.RUNNING.value, index=True)
    started_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Data
    input_data = Column(JSONType, nullable=True)
    output_data = Column(JSONType, nullable=True)
    data_size_bytes = Column(Integer, nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    error_type = Column(String(255), nullable=True)

    # Additional information (renamed from metadata to avoid conflicts)
    extra_metadata = Column(JSONType, nullable=True)

    # Audit fields
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    graph_execution = relationship("GraphExecution", back_populates="node_executions")

    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'skipped')", name="valid_node_status"),
        Index("idx_node_executions_execution_node", "execution_id", "node_name"),
        Index("idx_node_executions_status_started", "status", "started_at"),
        Index("idx_node_executions_run_id", "run_id"),
    )

    @validates('status')
    def validate_status(self, key: str, value: str) -> str:
        """Validate node execution status."""
        valid_statuses = {status.value for status in NodeStatus}
        if value not in valid_statuses:
            raise ValueError(f"Invalid status: {value}. Must be one of: {valid_statuses}")
        return value

    @property
    def is_running(self) -> bool:
        """Check if node is currently running."""
        return self.status == NodeStatus.RUNNING.value

    @property
    def is_completed(self) -> bool:
        """Check if node completed successfully."""
        return self.status == NodeStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if node failed."""
        return self.status == NodeStatus.FAILED.value

    def __repr__(self) -> str:
        """String representation of NodeExecution."""
        return f"<NodeExecution(id={self.id}, node_name={self.node_name}, status={self.status})>"


class ExecutionState(Base):
    """Model representing saved state snapshots during graph execution.

    This table stores state checkpoints that can be used for recovery
    in case of execution failures.
    """

    __tablename__ = "execution_states"

    # Primary key
    id = Column(String(36), primary_key=True, default=generate_uuid)

    # Foreign key to graph execution
    execution_id = Column(String(36), ForeignKey("graph_executions.id"), nullable=False, index=True)

    # Checkpoint information
    checkpoint_name = Column(String(255), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False, default=0, index=True)

    # State data
    state_data = Column(JSONType, nullable=False)
    state_size_bytes = Column(Integer, nullable=True)
    is_compressed = Column(Boolean, default=False)

    # Recovery information
    is_recovery_point = Column(Boolean, default=True, index=True)
    completed_nodes = Column(JSONType, nullable=True)  # List of completed node names
    pending_nodes = Column(JSONType, nullable=True)  # List of pending node names

    # Additional information (renamed from metadata to avoid conflicts)
    extra_metadata = Column(JSONType, nullable=True)

    # Audit fields
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)

    # Relationships
    graph_execution = relationship("GraphExecution", back_populates="execution_states")

    # Constraints
    __table_args__ = (
        Index("idx_execution_states_execution_checkpoint", "execution_id", "checkpoint_name"),
        Index("idx_execution_states_execution_sequence", "execution_id", "sequence_number"),
        Index("idx_execution_states_recovery", "execution_id", "is_recovery_point", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of ExecutionState."""
        return f"<ExecutionState(id={self.id}, checkpoint_name={self.checkpoint_name})>"