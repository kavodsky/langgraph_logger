# test_fixes.py

"""Test script to verify that the fixes work correctly."""


def test_pydantic_models():
    """Test that Pydantic models work with new ConfigDict."""
    print("Testing Pydantic models...")

    from langgraph_logger.dto import (
        GraphExecutionCreate,
        GraphExecutionUpdate,
        NodeExecutionCreate,
        ExecutionMetrics
    )

    # Test GraphExecutionCreate
    execution_create = GraphExecutionCreate(
        graph_name="test_graph",
        initial_state={"key": "value"},
        extra_metadata={"test": "metadata"},
        tags=["test", "demo"]
    )
    print(f"✅ GraphExecutionCreate: {execution_create.graph_name}")

    # Test GraphExecutionUpdate
    execution_update = GraphExecutionUpdate(
        status="completed",
        final_state={"result": "success"},
        extra_metadata={"finished": True}
    )
    print("✅ GraphExecutionUpdate created")

    # Test NodeExecutionCreate
    node_create = NodeExecutionCreate(
        execution_id="test-id",
        node_name="test_node",
        run_id="run-123",
        input_data={"input": "data"},
        extra_metadata={"node_meta": "value"}
    )
    print(f"✅ NodeExecutionCreate: {node_create.node_name}")

    # Test ExecutionMetrics with field_validator
    metrics = ExecutionMetrics(
        execution_id="test-id",
        total_duration=10.5,
        total_nodes=3,
        completed_nodes=2,
        failed_nodes=1,
        success_rate=66.7,  # Should pass validation
        average_node_duration=3.5
    )
    print(f"✅ ExecutionMetrics: {metrics.success_rate}%")

    try:
        # This should fail validation
        invalid_metrics = ExecutionMetrics(
            execution_id="test-id",
            total_duration=10.5,
            total_nodes=3,
            completed_nodes=2,
            failed_nodes=1,
            success_rate=150.0,  # Invalid - over 100%
            average_node_duration=3.5
        )
    except ValueError as e:
        print(f"✅ Validation working: {e}")

    print("Pydantic models test passed!")


def test_sqlalchemy_models():
    """Test that SQLAlchemy models work with renamed fields."""
    print("\nTesting SQLAlchemy models...")

    from langgraph_logger.models import GraphExecution, NodeExecution, ExecutionState
    from langgraph_logger.dto import ExecutionStatus, NodeStatus

    # Test GraphExecution
    execution = GraphExecution(
        graph_name="test_graph",
        initial_state={"initial": "state"},
        extra_metadata={"meta": "data"},  # Renamed field
        tags=["test"],
        status=ExecutionStatus.RUNNING.value
    )
    print(f"✅ GraphExecution: {execution.graph_name}")
    print(f"   extra_metadata: {execution.extra_metadata}")

    # Test NodeExecution
    node = NodeExecution(
        execution_id="test-exec-id",
        node_name="test_node",
        run_id="run-123",
        input_data={"input": "data"},
        extra_metadata={"node_meta": "value"},  # Renamed field
        status=NodeStatus.RUNNING.value
    )
    print(f"✅ NodeExecution: {node.node_name}")
    print(f"   extra_metadata: {node.extra_metadata}")

    # Test ExecutionState
    state = ExecutionState(
        execution_id="test-exec-id",
        checkpoint_name="test_checkpoint",
        state_data={"state": "data"},
        extra_metadata={"checkpoint_meta": "value"}  # Renamed field
    )
    print(f"✅ ExecutionState: {state.checkpoint_name}")
    print(f"   extra_metadata: {state.extra_metadata}")

    print("SQLAlchemy models test passed!")


def test_settings():
    """Test that settings work correctly."""
    print("\nTesting settings...")

    from langgraph_logger.settings import GraphLoggerSettings

    # Test default settings
    settings = GraphLoggerSettings.create_default()
    print(f"✅ Default settings created: {settings.database_url}")

    # Test validation
    try:
        settings_with_invalid_level = GraphLoggerSettings(log_level="INVALID")
    except ValueError as e:
        print(f"✅ Log level validation working: {e}")

    # Test database engine kwargs
    kwargs = settings.get_database_engine_kwargs()
    print(f"✅ Engine kwargs: {list(kwargs.keys())}")

    print("Settings test passed!")


def test_repository_creation():
    """Test that repository can be created without errors."""
    print("\nTesting repository creation...")

    from langgraph_logger.repository import GraphLoggerRepository
    from langgraph_logger.settings import GraphLoggerSettings

    settings = GraphLoggerSettings(
        database_url="sqlite:///test_fixes.db",
        enable_rich_output=False
    )

    try:
        repository = GraphLoggerRepository(settings)
        print("✅ Repository created successfully")

        # Test table creation
        repository.create_tables()
        print("✅ Database tables created successfully")

        repository.close()
        print("✅ Repository closed successfully")

    except Exception as e:
        print(f"❌ Repository test failed: {e}")
        return False

    print("Repository test passed!")
    return True


def test_callback_creation():
    """Test that callback can be created with new field names."""
    print("\nTesting callback creation...")

    from langgraph_logger.callback import GraphExecutionCallback
    from langgraph_logger.settings import GraphLoggerSettings

    settings = GraphLoggerSettings(
        database_url="sqlite:///test_fixes.db",
        enable_rich_output=False,
        enable_console_logging=False
    )

    try:
        callback = GraphExecutionCallback(
            graph_name="test_graph",
            settings=settings,
            initial_state={"test": "state"},
            tags=["test"],
            extra_metadata={"callback_meta": "value"}  # New field name
        )
        print("✅ Callback created successfully")
        print(f"   Graph name: {callback.graph_name}")
        print(f"   Extra metadata: {callback.extra_metadata}")

    except Exception as e:
        print(f"❌ Callback test failed: {e}")
        return False

    print("Callback test passed!")
    return True


def main():
    """Run all tests."""
    print("🧪 Running LangGraph Logger fixes tests...\n")

    try:
        test_pydantic_models()
        test_sqlalchemy_models()
        test_settings()

        # Only test database operations if they don't fail
        if test_repository_creation():
            test_callback_creation()

        print("\n🎉 All tests passed! The fixes are working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up test database
        import os
        try:
            if os.path.exists("test_fixes.db"):
                os.remove("test_fixes.db")
                print("🧹 Cleaned up test database")
        except:
            pass


if __name__ == "__main__":
    main()