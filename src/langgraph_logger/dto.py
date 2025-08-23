# src/langgraph_logger/dto.py

"""Pydantic data transfer objects for LangGraph Logger."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ExecutionStatus(str, Enum):
    """Enumeration of possible execution statuses."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NodeStatus(str, Enum):
    """Enumeration of possible node execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GraphExecutionCreate(BaseModel):
    """DTO for creating a new graph execution record."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    graph_name: str = Field(..., description="Name of the executed graph")
    initial_state: Dict[str, Any] = Field(..., description="Initial state of the graph")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    tags: Optional[List[str]] = Field(default=None, description="Tags for categorization")


class GraphExecutionUpdate(BaseModel):
    """DTO for updating a graph execution record."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    status: Optional[ExecutionStatus] = None
    final_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class GraphExecutionResponse(BaseModel):
    """DTO for graph execution response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: str
    graph_name: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    initial_state: Dict[str, Any]
    final_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    extra_metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class NodeExecutionCreate(BaseModel):
    """DTO for creating a node execution record."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    execution_id: str = Field(..., description="ID of the parent graph execution")
    node_name: str = Field(..., description="Name of the executed node")
    run_id: str = Field(..., description="Unique run ID for this node execution")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Node input data")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Node metadata")
    parent_run_id: Optional[str] = Field(default=None, description="Parent run ID for nested calls")


class NodeExecutionUpdate(BaseModel):
    """DTO for updating a node execution record."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    status: Optional[NodeStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class NodeExecutionResponse(BaseModel):
    """DTO for node execution response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: str
    execution_id: str
    node_name: str
    run_id: str
    status: NodeStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    parent_run_id: Optional[str] = None


class ExecutionStateCreate(BaseModel):
    """DTO for creating an execution state snapshot."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    execution_id: str = Field(..., description="ID of the parent graph execution")
    checkpoint_name: str = Field(..., description="Name of the checkpoint")
    state_data: Dict[str, Any] = Field(..., description="Current state data")
    extra_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Checkpoint metadata")


class ExecutionStateResponse(BaseModel):
    """DTO for execution state response."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: str
    execution_id: str
    checkpoint_name: str
    state_data: Dict[str, Any]
    created_at: datetime
    extra_metadata: Optional[Dict[str, Any]] = None


class ExecutionMetrics(BaseModel):
    """DTO for execution metrics and statistics."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    execution_id: str
    total_duration: float
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    success_rate: float
    average_node_duration: float
    longest_node: Optional[str] = None
    longest_node_duration: Optional[float] = None
    shortest_node: Optional[str] = None
    shortest_node_duration: Optional[float] = None
    parallel_execution_peak: int = 0
    error_summary: List[Dict[str, Any]] = Field(default_factory=list)
    node_timings: Dict[str, Dict[str, Union[float, int]]] = Field(default_factory=dict)

    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v: float) -> float:
        """Validate success rate is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Success rate must be between 0 and 100")
        return v


class RecoveryInfo(BaseModel):
    """DTO for graph execution recovery information."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    execution_id: str
    last_successful_checkpoint: str
    available_checkpoints: List[str]
    recovery_state: Dict[str, Any]
    failed_nodes: List[str]
    can_recover: bool
    recovery_instructions: Optional[str] = None