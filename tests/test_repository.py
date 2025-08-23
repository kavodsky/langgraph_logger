"""Test repository functionality."""

import pytest
from datetime import datetime

from langgraph_logger.dto import (
    ExecutionStatus, NodeStatus, GraphExecutionCreate,
    NodeExecutionCreate, ExecutionStateCreate
)


def test_create_graph_execution(test_repository):
    """Test creating a graph execution."""
    data = GraphExecutionCreate(
        graph_name="test_graph",
        initial_state={"input": "test"},
        tags=["test"],
        metadata={"test": True}
    )

    execution_id = test_repository.create_graph_execution(data)
    assert execution_id is not None

    execution = test_repository.get_graph_execution(execution_id)
    assert execution is not None
    assert execution.graph_name == "test_graph"
    assert execution.status == ExecutionStatus.RUNNING.value


def test_node_execution_lifecycle(test_repository):
    """Test complete node execution lifecycle."""
    # Create graph execution first
    graph_data = GraphExecutionCreate(
        graph_name="test_graph",
        initial_state={"input": "test"}
    )
    execution_id = test_repository.create_graph_execution(graph_data)

    # Create node execution
    node_data = NodeExecutionCreate(
        execution_id=execution_id,
        node_name="test_node",
        run_id="test_run_123",
        input_data={"test": "input"}
    )

    node_id = test_repository.create_node_execution(node_data)
    assert node_id is not None

    # Update node execution
    from langgraph_logger.dto import NodeExecutionUpdate
    update_data = NodeExecutionUpdate(
        status=NodeStatus.COMPLETED,
        output_data={"result": "success"}
    )

    success = test_repository.update_node_execution("test_run_123", update_data)
    assert success is True


def test_execution_states(test_repository):
    """Test execution state management."""
    # Create graph execution
    graph_data = GraphExecutionCreate(
        graph_name="test_graph",
        initial_state={"input": "test"}
    )
    execution_id = test_repository.create_graph_execution(graph_data)

    # Create state snapshot
    state_data = ExecutionStateCreate(
        execution_id=execution_id,
        checkpoint_name="test_checkpoint",
        state_data={"current": "state"},
        metadata={"checkpoint": True}
    )

    state_id = test_repository.create_execution_state(state_data)
    assert state_id is not None

    # Get latest state
    latest_state = test_repository.get_latest_execution_state(execution_id)
    assert latest_state is not None
    assert latest_state.checkpoint_name == "test_checkpoint"


def test_execution_metrics(test_repository):
    """Test execution metrics calculation."""
    # Create a complete execution scenario
    graph_data = GraphExecutionCreate(
        graph_name="metrics_test",
        initial_state={"input": "test"}
    )
    execution_id = test_repository.create_graph_execution(graph_data)

    # Add some node executions
    for i in range(3):
        node_data = NodeExecutionCreate(
            execution_id=execution_id,
            node_name=f"node_{i + 1}",
            run_id=f"run_{i + 1}",
            input_data={"step": i + 1}
        )
        test_repository.create_node_execution(node_data)

        # Complete the node
        from langgraph_logger.dto import NodeExecutionUpdate
        update_data = NodeExecutionUpdate(status=NodeStatus.COMPLETED)
        test_repository.update_node_execution(f"run_{i + 1}", update_data)

    # Update execution stats
    test_repository.update_execution_stats(execution_id)

    # Get metrics
    metrics = test_repository.get_execution_metrics(execution_id)
    assert metrics is not None
    assert metrics.total_nodes == 3
    assert metrics.completed_nodes == 3
    assert metrics.success_rate == 100.0