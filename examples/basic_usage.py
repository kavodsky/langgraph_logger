# examples/basic_usage.py

"""Basic usage examples for LangGraph Logger."""
import time

from langgraph.graph import StateGraph
from langgraph_logger import GraphExecutionCallback, GraphLoggerSettings
from langgraph_logger.utils import logged_graph_execution, create_logger_callback
from typing import Dict, Any


def basic_callback_usage():
    """Example of basic callback usage."""

    # Create a simple graph (placeholder)
    def simple_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": state.get("input", 0) * 2}

    # Setup callback with default settings
    callback = create_logger_callback(
        graph_name="simple_example",
        initial_state={"input": 5},
        tags=["example", "basic"],
        metadata={"version": "1.0", "environment": "development"}
    )

    # In practice, you would use this with an actual LangGraph
    # graph_result = graph.invoke(initial_state, config={"callbacks": [callback]})

    # Manually finalize for demo
    callback.finalize_execution(final_state={"result": 10})

    # Print summary
    callback.print_execution_summary()

    return callback.execution_id


def context_manager_usage():
    """Example using context manager for automatic cleanup."""

    initial_state = {"data": [1, 2, 3, 4, 5]}

    with logged_graph_execution(
            graph_name="context_example",
            initial_state=initial_state,
            tags=["example", "context"],
            enable_rich_output=True
    ) as callback:
        # Simulate graph execution
        # In practice: result = graph.invoke(initial_state, config={"callbacks": [callback]})

        # For demo, manually trigger callbacks
        callback.on_chain_start(
            {"name": "process_data"},
            initial_state,
            run_id="run_1"
        )

        # Simulate processing
        import time
        time.sleep(0.1)

        callback.on_chain_end(
            {"processed_data": [x * 2 for x in initial_state["data"]]},
            run_id="run_1"
        )

        # Context manager will handle finalization automatically
        return callback.execution_id


def custom_settings_usage():
    """Example with custom settings."""

    # Create custom settings
    settings = GraphLoggerSettings(
        database_url="sqlite:///./custom_executions.db",
        enable_rich_output=True,
        auto_save_state=True,
        recovery_checkpoint_interval=3,  # Checkpoint every 3 nodes
        log_level="DEBUG"
    )

    callback = GraphExecutionCallback(
        graph_name="custom_settings_example",
        settings=settings,
        initial_state={"custom": True},
        metadata={"custom_setting": True}
    )

    # Simulate multiple nodes
    nodes = ["extract", "transform", "validate", "load", "notify"]

    for i, node_name in enumerate(nodes):
        run_id = f"run_{i + 1}"

        callback.on_chain_start(
            {"name": node_name},
            {"step": i + 1, "node": node_name},
            run_id=run_id
        )

        time.sleep(0.05)  # Simulate work

        if node_name == "validate" and i == 2:
            # Simulate an error
            callback.on_chain_error(
                ValueError(f"Validation failed for step {i + 1}"),
                run_id=run_id
            )
        else:
            callback.on_chain_end(
                {"completed": node_name, "step": i + 1},
                run_id=run_id
            )

    callback.finalize_execution(final_state={"completed_nodes": len(nodes) - 1})
    callback.print_execution_summary()

    return callback.execution_id


def parallel_execution_simulation():
    """Example simulating parallel node execution."""

    callback = create_logger_callback(
        graph_name="parallel_example",
        initial_state={"batch_size": 3},
        metadata={"parallel": True}
    )

    import threading
    import time
    import random

    def simulate_node(node_name: str, run_id: str, duration: float):
        """Simulate a node execution in a thread."""
        callback.on_chain_start(
            {"name": node_name},
            {"node": node_name, "duration": duration},
            run_id=run_id
        )

        time.sleep(duration)

        # Randomly fail some nodes
        if random.random() < 0.1:  # 10% failure rate
            callback.on_chain_error(
                RuntimeError(f"Random failure in {node_name}"),
                run_id=run_id
            )
        else:
            callback.on_chain_end(
                {"result": f"completed_{node_name}", "duration": duration},
                run_id=run_id
            )

    # Start multiple nodes in parallel
    threads = []
    for i in range(5):
        node_name = f"worker_{i + 1}"
        run_id = f"parallel_run_{i + 1}"
        duration = random.uniform(0.1, 0.5)

        thread = threading.Thread(
            target=simulate_node,
            args=(node_name, run_id, duration)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    callback.finalize_execution(final_state={"parallel_completed": True})
    callback.print_execution_summary()

    return callback.execution_id


def recovery_example():
    """Example demonstrating recovery functionality."""
    from langgraph_logger.utils import ExecutionRecover

    # First, create a failed execution
    callback = create_logger_callback(
        graph_name="recovery_demo",
        initial_state={"step": 1},
        auto_save_state=True
    )

    # Simulate some successful steps with checkpoints
    steps = ["init", "prepare", "process", "validate", "finalize"]

    for i, step in enumerate(steps):
        run_id = f"step_{i + 1}"

        callback.on_chain_start(
            {"name": step},
            {"current_step": i + 1, "data": f"step_{i + 1}_data"},
            run_id=run_id
        )

        # Create manual checkpoints for important steps
        if step in ["prepare", "process"]:
            callback.create_manual_checkpoint(
                f"checkpoint_after_{step}",
                {"completed_step": step, "step_number": i + 1}
            )

        time.sleep(0.02)

        # Simulate failure at validation step
        if step == "validate":
            callback.on_chain_error(
                ValueError("Validation failed - data corrupted"),
                run_id=run_id
            )
            break
        else:
            callback.on_chain_end(
                {"completed": step, "step": i + 1},
                run_id=run_id
            )

    # Finalize as failed
    callback.finalize_execution(error=ValueError("Execution failed at validation"))

    execution_id = callback.execution_id
    print(f"Created failed execution: {execution_id}")

    # Now demonstrate recovery
    recoverer = ExecutionRecover()

    if recoverer.can_recover(execution_id):
        print("✅ Execution can be recovered!")

        # List available checkpoints
        checkpoints = recoverer.list_checkpoints(execution_id)
        print(f"Available checkpoints: {checkpoints}")

        # Get recovery state from the last good checkpoint
        recovery_state = recoverer.get_recovery_state(execution_id, "checkpoint_after_process")
        print(f"Recovery state: {recovery_state}")

        # In practice, you would use this state to restart the graph
        print("📝 To recover: restart graph with the recovery state from 'checkpoint_after_process'")
    else:
        print("❌ Execution cannot be recovered")

    recoverer.close()
    return execution_id


if __name__ == "__main__":
    print("🚀 Running LangGraph Logger Examples")
    print("=" * 50)

    print("\n1. Basic callback usage:")
    execution_id_1 = basic_callback_usage()

    print(f"\n2. Context manager usage:")
    execution_id_2 = context_manager_usage()

    print(f"\n3. Custom settings usage:")
    execution_id_3 = custom_settings_usage()

    print(f"\n4. Parallel execution simulation:")
    execution_id_4 = parallel_execution_simulation()

    print(f"\n5. Recovery example:")
    execution_id_5 = recovery_example()

    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print(f"📊 Created {5} example executions")
    print(f"💾 Check your database for execution records")
    print(f"🔍 Use CLI: langgraph-logger list --limit 10")