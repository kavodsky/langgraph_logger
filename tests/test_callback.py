"""Test callback handler."""

import pytest
import time
from threading import Thread

from langgraph_logger.callback import GraphExecutionCallback


def test_callback_initialization(test_settings):
    """Test callback initialization."""
    callback = GraphExecutionCallback(
        graph_name="test_callback",
        settings=test_settings,
        initial_state={"test": True}
    )

    assert callback.graph_name == "test_callback"
    assert callback.initial_state == {"test": True}


def test_node_execution_flow(test_settings):
    """Test complete node execution flow."""
    callback = GraphExecutionCallback(
        graph_name="test_flow",
        settings=test_settings,
        initial_state={"input": "test"}
    )

    # Start node
    callback.on_chain_start(
        {"name": "test_node"},
        {"input_data": "test"},
        run_id="test_run"
    )

    assert "test_run" in callback.active_runs
    assert callback.total_nodes_started == 1

    # End node
    callback.on_chain_end(
        {"output_data": "result"},
        run_id="test_run"
    )

    assert "test_run" not in callback.active_runs
    assert "test_run" in callback.completed_runs
    assert callback.total_nodes_completed == 1


def test_parallel_execution_tracking(test_settings):
    """Test parallel execution tracking."""
    callback = GraphExecutionCallback(
        graph_name="parallel_test",
        settings=test_settings,
        initial_state={"batch": True}
    )

    def start_node(run_id: str):
        callback.on_chain_start(
            {"name": f"node_{run_id}"},
            {"parallel": True},
            run_id=run_id
        )
        time.sleep(0.01)  # Simulate work
        callback.on_chain_end(
            {"result": f"done_{run_id}"},
            run_id=run_id
        )

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = Thread(target=start_node, args=(f"parallel_run_{i}",))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    assert callback.max_concurrent_nodes >= 1
    assert callback.total_nodes_completed == 3


def test_error_handling(test_settings):
    """Test error handling in callback."""
    callback = GraphExecutionCallback(
        graph_name="error_test",
        settings=test_settings,
        initial_state={"test": True}
    )

    # Start node
    callback.on_chain_start(
        {"name": "failing_node"},
        {"input": "test"},
        run_id="error_run"
    )

    # Simulate error
    test_error = ValueError("Test error message")
    callback.on_chain_error(test_error, run_id="error_run")

    assert "error_run" not in callback.active_runs
    assert "error_run" in callback.failed_runs
    assert callback.total_nodes_failed == 1


def test_manual_checkpoints(test_settings):
    """Test manual checkpoint creation."""
    callback = GraphExecutionCallback(
        graph_name="checkpoint_test",
        settings=test_settings,
        initial_state={"step": 1}
    )

    # Create manual checkpoint
    callback.create_manual_checkpoint("test_checkpoint", {"step": 2})

    assert callback.checkpoint_counter == 1


def test_context_manager(test_settings):
    """Test callback as context manager."""
    with GraphExecutionCallback(
            graph_name="context_test",
            settings=test_settings,
            initial_state={"context": True}
    ) as callback:
        # Simulate some work
        callback.on_chain_start(
            {"name": "context_node"},
            {"test": True},
            run_id="context_run"
        )
        callback.on_chain_end(
            {"result": True},
            run_id="context_run"
        )

    # After context exit, execution should be finalized
    summary = callback.get_execution_summary()
    assert summary["total_nodes_completed"] == 1
