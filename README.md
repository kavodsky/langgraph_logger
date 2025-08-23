# LangGraph Logger

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced logging and state management for LangGraph executions with database persistence, parallel execution tracking, and automatic recovery capabilities.

## Features

🔍 **Comprehensive Logging**
- Database persistence of all execution logs and metrics
- Support for SQLite, PostgreSQL, and MySQL
- Rich console output with progress tracking
- Detailed node-level execution tracking

⚡ **Parallel Execution Support**
- Thread-safe tracking of concurrent node executions
- Real-time monitoring of parallel execution peaks
- Accurate timing and resource usage metrics

🔄 **Automatic Recovery**
- Smart state checkpointing during execution
- Recovery state snapshots for failed executions
- Configurable checkpoint intervals
- Manual checkpoint creation support

📊 **Advanced Analytics**
- Comprehensive execution metrics and statistics
- Node performance analysis and bottleneck detection
- Success rates and error pattern analysis
- Export/import functionality for execution data

🛠️ **Developer Friendly**
- Clean, documented Python API
- Command-line interface for management
- Context managers for easy integration
- Configurable settings with sensible defaults

## Installation

### From PyPI (when published)

```bash
pip install langgraph-logger
```

### Development Installation with UV

```bash
# Clone the repository
git clone https://github.com/yourusername/langgraph-logger.git
cd langgraph-logger

# Install with uv
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from langgraph_logger import create_logger_callback

# Create a callback for your graph
callback = create_logger_callback(
    graph_name="my_workflow",
    initial_state={"input": "data"},
    tags=["production", "v1.0"],
    metadata={"user_id": "12345"}
)

# Use with your LangGraph
result = graph.invoke(
    initial_state,
    config={"callbacks": [callback]}
)

# Finalize and view summary
callback.finalize_execution(final_state=result)
callback.print_execution_summary()
```

### Context Manager (Recommended)

```python
from langgraph_logger.utils import logged_graph_execution

initial_state = {"query": "What is the weather?"}

with logged_graph_execution(
    graph_name="weather_agent", 
    initial_state=initial_state
) as callback:
    result = graph.invoke(
        initial_state, 
        config={"callbacks": [callback]}
    )
    # Automatic cleanup and summary printing
```

## Configuration

### Environment Variables

```bash
# Database configuration
export LANGGRAPH_LOGGER_DATABASE_URL="sqlite:///./executions.db"
export LANGGRAPH_LOGGER_DATABASE_ECHO=false

# Logging configuration
export LANGGRAPH_LOGGER_LOG_LEVEL="INFO"
export LANGGRAPH_LOGGER_ENABLE_RICH_OUTPUT=true

# State management
export LANGGRAPH_LOGGER_AUTO_SAVE_STATE=true
export LANGGRAPH_LOGGER_RECOVERY_CHECKPOINT_INTERVAL=5

# Performance settings
export LANGGRAPH_LOGGER_CLEANUP_OLD_EXECUTIONS_DAYS=30
export LANGGRAPH_LOGGER_BATCH_INSERT_SIZE=100
```

### Custom Settings

```python
from langgraph_logger import GraphLoggerSettings, GraphExecutionCallback

# Create custom settings
settings = GraphLoggerSettings(
    database_url="postgresql://user:pass@localhost/langgraph_logs",
    enable_rich_output=True,
    auto_save_state=True,
    recovery_checkpoint_interval=3,
    log_level="DEBUG",
    cleanup_old_executions_days=7
)

# Use custom settings
callback = GraphExecutionCallback(
    graph_name="custom_workflow",
    settings=settings,
    initial_state=initial_state
)
```

## Database Support

### SQLite (Default)
```python
settings.database_url = "sqlite:///./langgraph_executions.db"
```

### PostgreSQL
```bash
pip install langgraph-logger[postgres]
```
```python
settings.database_url = "postgresql://user:password@localhost:5432/langgraph_db"
```

### MySQL
```bash
pip install langgraph-logger[mysql]
```
```python
settings.database_url = "mysql+pymysql://user:password@localhost:3306/langgraph_db"
```

## Command Line Interface

### Initialize Database
```bash
# Initialize with default SQLite database
langgraph-logger init

# Initialize with custom database
langgraph-logger init --db "postgresql://user:pass@localhost/db"

# Force recreate tables
langgraph-logger init --force
```

### List Executions
```bash
# List recent executions
langgraph-logger list

# Filter by graph name
langgraph-logger list --graph "my_workflow"

# Filter by status
langgraph-logger list --status "failed"

# Show executions from last 7 days
langgraph-logger list --days 7 --limit 20
```

### Show Execution Details
```bash
# Basic execution info
langgraph-logger show abc123def

# Include node details
langgraph-logger show abc123def --nodes

# Include saved states
langgraph-logger show abc123def --states

# Include detailed metrics
langgraph-logger show abc123def --metrics
```

### Recovery Information
```bash
# Show recovery options for failed execution
langgraph-logger recovery abc123def
```

### Statistics and Analytics
```bash
# Show execution statistics
langgraph-logger stats

# Stats for specific graph
langgraph-logger stats --graph "my_workflow"

# Stats for last 30 days
langgraph-logger stats --days 30
```

### Cleanup Old Data
```bash
# Clean up executions older than 30 days
langgraph-logger cleanup --days 30

# Dry run to see what would be deleted
langgraph-logger cleanup --days 30 --dry-run

# Skip confirmation prompt
langgraph-logger cleanup --days 30 --confirm
```

## Advanced Features

### Manual Checkpoints

```python
callback = create_logger_callback("workflow_with_checkpoints", initial_state)

# Create manual checkpoints at important points
callback.create_manual_checkpoint(
    "before_critical_operation", 
    current_state
)

# Execute critical operation
result = critical_operation(current_state)

callback.create_manual_checkpoint(
    "after_critical_operation", 
    result
)
```

### Recovery from Failures

```python
from langgraph_logger.utils import ExecutionRecover

# Check if execution can be recovered
recoverer = ExecutionRecover()

if recoverer.can_recover(failed_execution_id):
    # Get list of available checkpoints
    checkpoints = recoverer.list_checkpoints(failed_execution_id)
    print(f"Available checkpoints: {checkpoints}")
    
    # Get recovery state from specific checkpoint
    recovery_state = recoverer.get_recovery_state(
        failed_execution_id, 
        "before_critical_operation"
    )
    
    # Restart execution from recovery state
    with logged_graph_execution("recovered_workflow", recovery_state) as callback:
        result = graph.invoke(recovery_state, config={"callbacks": [callback]})

recoverer.close()
```

### Export and Analysis

```python
from langgraph_logger.utils import export_execution_data, get_execution_summary

# Export execution data to JSON
export_execution_data(
    execution_id="abc123def",
    output_file="execution_analysis.json",
    include_states=True,
    include_nodes=True
)

# Get comprehensive execution summary
summary = get_execution_summary(execution_id)
if summary:
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Total duration: {summary['duration']:.2f}s")
    print(f"Can recover: {summary['can_recover']}")
```

### Parallel Execution Tracking

The logger automatically handles parallel node executions:

```python
# The callback safely tracks multiple concurrent nodes
callback = create_logger_callback("parallel_workflow", initial_state)

# LangGraph will automatically handle parallel execution
# The callback tracks each node with unique run_ids
result = parallel_graph.invoke(
    initial_state,
    config={"callbacks": [callback]}
)

# View parallel execution statistics
summary = callback.get_execution_summary()
print(f"Max concurrent nodes: {summary['max_concurrent_nodes']}")
```

## Integration Examples

### With Your Existing LangGraph

```python
from langgraph.graph import StateGraph
from langgraph_logger.utils import logged_graph_execution

# Your existing graph setup
graph = StateGraph(YourState)
graph.add_node("node1", node1_function)
graph.add_node("node2", node2_function)
# ... add more nodes and edges
compiled_graph = graph.compile()

# Add logging with minimal changes
initial_state = {"input": "your_data"}

with logged_graph_execution("your_graph", initial_state) as callback:
    result = compiled_graph.invoke(
        initial_state,
        config={"callbacks": [callback]}
    )
```

### Custom Metadata and Tagging

```python
callback = create_logger_callback(
    graph_name="customer_service_bot",
    initial_state=initial_state,
    tags=["production", "customer-service", "v2.1"],
    metadata={
        "user_id": "user_12345",
        "session_id": "session_abc123",
        "deployment": "us-west-2",
        "model_version": "gpt-4",
        "priority": "high"
    }
)
```

## Monitoring and Alerting

### Custom Metrics Collection

```python
from langgraph_logger import GraphLoggerRepository

repository = GraphLoggerRepository(settings)

# Get metrics for analysis
metrics = repository.get_execution_metrics(execution_id)
if metrics:
    # Alert on low success rate
    if metrics.success_rate < 90.0:
        send_alert(f"Low success rate: {metrics.success_rate}%")
    
    # Alert on slow executions
    if metrics.total_duration > 300:  # 5 minutes
        send_alert(f"Slow execution: {metrics.total_duration}s")
    
    # Alert on high error rate
    error_rate = (metrics.failed_nodes / metrics.total_nodes) * 100
    if error_rate > 10.0:
        send_alert(f"High error rate: {error_rate}%")

repository.close()
```

## Performance Considerations

### Database Optimization

- Use PostgreSQL for production environments with high throughput
- Configure appropriate connection pooling settings
- Set up database indices for frequently queried fields
- Use cleanup scheduling to manage database size

### Memory Usage

- Large state objects are automatically truncated in logs
- Enable compression for large state snapshots
- Configure reasonable checkpoint intervals
- Monitor state size warnings

### Concurrent Executions

- The logger is thread-safe and handles parallel executions
- Database connections are properly managed
- Lock contention is minimized through efficient design

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Test database connection
   python -c "from langgraph_logger.utils import validate_database_connection; print(validate_database_connection('your_db_url'))"
   ```

2. **Large State Objects**
   ```python
   # Enable state size monitoring
   settings.max_state_size_mb = 10  # Warn if state > 10MB
   ```

3. **Performance Issues**
   ```python
   # Tune performance settings
   settings.batch_insert_size = 50  # Smaller batches
   settings.database_pool_size = 10  # More connections
   ```

### Debug Mode

```python
# Enable debug logging
settings.log_level = "DEBUG"
settings.database_echo = True  # Log SQL queries
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run tests: `pytest`
5. Submit a pull request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/langgraph-logger.git
cd langgraph-logger

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### Version 0.1.0
- Initial release
- Database persistence for executions and states
- Parallel execution tracking
- Recovery and checkpoint functionality
- Rich console output
- Command-line interface
- Export/import capabilities

## Support

- 📖 [Documentation](https://github.com/yourusername/langgraph-logger#readme)
- 🐛 [Issue Tracker](https://github.com/yourusername/langgraph-logger/issues)
- 💬 [Discussions](https://github.com/yourusername/langgraph-logger/discussions)

## Acknowledgments

- Built for [LangGraph](https://langchain-ai.github.io/langgraph/) workflows
- Uses [SQLAlchemy](https://sqlalchemy.org/) for database operations
- Rich console output powered by [Rich](https://rich.readthedocs.io/)
- CLI built with [Typer](https://typer.tiangolo.com/)