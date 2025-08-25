# src/langgraph_logger/cli.py

"""Command-line interface for LangGraph Logger with enhanced error handling."""

import traceback
from datetime import datetime, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from .repository import GraphLoggerRepository
from .settings import GraphLoggerSettings
from .dto import ExecutionStatus

app = typer.Typer(
    name="langgraph-logger",
    help="LangGraph Logger CLI - Manage and view graph execution logs"
)
console = Console()


def handle_error(error: Exception, operation: str):
    """Handle and display errors with full tracebacks."""
    console.print(f"❌ [red]Error during {operation}:[/red]")
    console.print(f"   {type(error).__name__}: {str(error)}")

    # Show traceback in a collapsible panel
    tb_text = traceback.format_exc()
    console.print(Panel(
        Syntax(tb_text, "python", theme="monokai", line_numbers=True),
        title="Full Traceback",
        expand=False,
        border_style="red"
    ))


@app.command("init")
def init_database(
        database_url: Optional[str] = typer.Option(None, "--db", help="Database URL"),
        force: bool = typer.Option(False, "--force", help="Force recreation of tables"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """Initialize the database tables."""
    try:
        settings = GraphLoggerSettings.create_default()
        if database_url:
            settings.database_url = database_url

        console.print(f"🔧 Initializing database: {settings.database_url}")
        repository = GraphLoggerRepository(settings)

        if force:
            with console.status("Dropping existing tables..."):
                repository.drop_tables()
                console.print("✅ [green]Existing tables dropped[/green]")

        with console.status("Creating database tables..."):
            repository.create_tables()
            console.print("✅ [green]Database tables created successfully[/green]")

        # Test basic functionality
        with console.status("Testing database connection..."):
            # Try a simple query to verify everything works
            executions = repository.list_graph_executions(limit=1)
            console.print("✅ [green]Database connection verified[/green]")

    except Exception as e:
        if show_traceback:
            handle_error(e, "database initialization")
        else:
            console.print(f"❌ [red]Error initializing database: {e}[/red]")
            console.print("💡 [yellow]Use --traceback to see full error details[/yellow]")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("list")
def list_executions(
        graph_name: Optional[str] = typer.Option(None, "--graph", help="Filter by graph name"),
        status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
        limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
        days: Optional[int] = typer.Option(None, "--days", help="Show executions from last N days"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """List graph executions."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = ExecutionStatus(status.lower())
            except ValueError:
                console.print(f"❌ [red]Invalid status: {status}[/red]")
                console.print(f"Valid statuses: {', '.join([s.value for s in ExecutionStatus])}")
                raise typer.Exit(1)

        # Calculate date filter
        start_date = None
        if days:
            start_date = datetime.utcnow() - timedelta(days=days)

        with console.status("Fetching executions..."):
            executions = repository.list_graph_executions(
                graph_name=graph_name,
                status=status_filter,
                limit=limit,
                start_date=start_date
            )

        if not executions:
            console.print("🔭 [yellow]No executions found[/yellow]")
            return

        # Create table
        table = Table(title="Graph Executions")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Graph", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Started", style="magenta")
        table.add_column("Duration", style="yellow")
        table.add_column("Nodes", style="white")
        table.add_column("Success Rate", style="green")

        for execution in executions:
            # Format duration
            duration = "N/A"
            if execution.duration_seconds:
                duration = f"{execution.duration_seconds:.1f}s"

            # Format success rate
            success_rate = f"{execution.success_rate:.1f}%"

            # Format node info
            nodes_info = f"{execution.completed_nodes}/{execution.total_nodes}"
            if execution.failed_nodes > 0:
                nodes_info += f" ({execution.failed_nodes} failed)"

            # Status color
            status_style = "green" if execution.is_completed else "red" if execution.is_failed else "yellow"

            table.add_row(
                execution.id[:8],
                execution.graph_name,
                f"[{status_style}]{execution.status.value}[/{status_style}]",
                execution.started_at.strftime("%Y-%m-%d %H:%M"),
                duration,
                nodes_info,
                success_rate
            )

        console.print(table)

    except Exception as e:
        if show_traceback:
            handle_error(e, "listing executions")
        else:
            console.print(f"❌ [red]Error listing executions: {e}[/red]")
            console.print("💡 [yellow]Use --traceback to see full error details[/yellow]")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("show")
def show_execution(
        execution_id: str = typer.Argument(..., help="Execution ID to show"),
        nodes: bool = typer.Option(False, "--nodes", help="Show node details"),
        states: bool = typer.Option(False, "--states", help="Show saved states"),
        metrics: bool = typer.Option(False, "--metrics", help="Show detailed metrics"),
        errors: bool = typer.Option(False, "--errors", help="Show error details"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """Show detailed information about a specific execution."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        with console.status("Fetching execution details..."):
            # Get execution
            execution = repository.get_graph_execution(execution_id)
            if not execution:
                console.print(f"❌ [red]Execution not found: {execution_id}[/red]")
                raise typer.Exit(1)

        # Main execution info
        table = Table(title=f"Execution Details: {execution.id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("ID", execution.id)
        table.add_row("Graph Name", execution.graph_name)
        table.add_row("Status",
                      f"[{'green' if execution.is_completed else 'red' if execution.is_failed else 'yellow'}]{execution.status.value}[/]")
        table.add_row("Started", execution.started_at.strftime("%Y-%m-%d %H:%M:%S"))

        if execution.completed_at:
            table.add_row("Completed", execution.completed_at.strftime("%Y-%m-%d %H:%M:%S"))

        if execution.duration_seconds:
            table.add_row("Duration", f"{execution.duration_seconds:.3f}s")

        table.add_row("Total Nodes", str(execution.total_nodes))
        table.add_row("Completed Nodes", str(execution.completed_nodes))
        table.add_row("Failed Nodes", str(execution.failed_nodes))
        table.add_row("Success Rate", f"{execution.success_rate:.1f}%")

        if execution.error_message:
            error_preview = execution.error_message[:100] + "..." if len(
                execution.error_message) > 100 else execution.error_message
            table.add_row("Error", f"[red]{error_preview}[/red]")

        console.print(table)

        # Show full error details if requested
        if errors and execution.error_message:
            console.print("\n")
            error_panel = Panel(
                execution.error_message,
                title="Error Message",
                border_style="red"
            )
            console.print(error_panel)

            if execution.error_traceback:
                tb_panel = Panel(
                    Syntax(execution.error_traceback, "python", theme="monokai", line_numbers=True),
                    title="Error Traceback",
                    border_style="red"
                )
                console.print(tb_panel)

        # Show nodes if requested
        if nodes:
            with console.status("Fetching node details..."):
                node_executions = repository.get_node_executions_for_graph(execution_id)

            if node_executions:
                console.print("\n")
                nodes_table = Table(title="Node Executions")
                nodes_table.add_column("Node Name", style="blue")
                nodes_table.add_column("Status", style="green")
                nodes_table.add_column("Duration", style="yellow")
                nodes_table.add_column("Started", style="magenta")
                nodes_table.add_column("Error", style="red")

                for node in node_executions:
                    duration = f"{node.duration_seconds:.3f}s" if node.duration_seconds else "N/A"
                    status_style = "green" if node.is_completed else "red" if node.is_failed else "yellow"
                    error_info = node.error_message[:30] + "..." if node.error_message and len(
                        node.error_message) > 30 else (node.error_message or "")

                    nodes_table.add_row(
                        node.node_name,
                        f"[{status_style}]{node.status.value}[/{status_style}]",
                        duration,
                        node.started_at.strftime("%H:%M:%S"),
                        f"[red]{error_info}[/red]" if error_info else ""
                    )

                console.print(nodes_table)

        # Show states if requested
        if states:
            with console.status("Fetching saved states..."):
                recovery_states = repository.get_recovery_states(execution_id)

            if recovery_states:
                console.print("\n")
                states_table = Table(title="Saved States")
                states_table.add_column("Checkpoint", style="blue")
                states_table.add_column("Created", style="magenta")
                states_table.add_column("Size (KB)", style="yellow")

                for state in recovery_states:
                    size_kb = len(str(state.state_data)) / 1024
                    states_table.add_row(
                        state.checkpoint_name,
                        state.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{size_kb:.1f}"
                    )

                console.print(states_table)

        # Show metrics if requested
        if metrics:
            with console.status("Calculating metrics..."):
                execution_metrics = repository.get_execution_metrics(execution_id)

            if execution_metrics:
                console.print("\n")
                metrics_table = Table(title="Detailed Metrics")
                metrics_table.add_column("Node", style="blue")
                metrics_table.add_column("Executions", style="cyan")
                metrics_table.add_column("Total Time", style="yellow")
                metrics_table.add_column("Avg Time", style="green")
                metrics_table.add_column("Min/Max", style="magenta")

                for node_name, stats in execution_metrics.node_timings.items():
                    metrics_table.add_row(
                        node_name,
                        str(stats['executions']),
                        f"{stats['total_time']:.3f}s",
                        f"{stats['avg_time']:.3f}s",
                        f"{stats['min_time']:.3f}s / {stats['max_time']:.3f}s"
                    )

                console.print(metrics_table)

    except Exception as e:
        if show_traceback:
            handle_error(e, "showing execution details")
        else:
            console.print(f"❌ [red]Error showing execution: {e}[/red]")
            console.print("💡 [yellow]Use --traceback to see full error details[/yellow]")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("recovery")
def show_recovery_info(
        execution_id: str = typer.Argument(..., help="Execution ID to show recovery info for"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """Show recovery information for a failed execution."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        with console.status("Fetching recovery information..."):
            recovery_info = repository.get_recovery_info(execution_id)

        if not recovery_info:
            console.print(f"❌ [red]No recovery info found for execution: {execution_id}[/red]")
            raise typer.Exit(1)

        # Recovery info table
        table = Table(title=f"Recovery Information: {execution_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Can Recover", "✅ Yes" if recovery_info.can_recover else "❌ No")
        table.add_row("Last Checkpoint", recovery_info.last_successful_checkpoint)
        table.add_row("Available Checkpoints", str(len(recovery_info.available_checkpoints)))
        table.add_row("Failed Nodes", ", ".join(recovery_info.failed_nodes))

        if recovery_info.recovery_instructions:
            table.add_row("Instructions", recovery_info.recovery_instructions)

        console.print(table)

        # List available checkpoints
        if recovery_info.available_checkpoints:
            console.print("\n")
            checkpoints_table = Table(title="Available Checkpoints")
            checkpoints_table.add_column("Checkpoint Name", style="blue")

            for checkpoint in recovery_info.available_checkpoints:
                checkpoints_table.add_row(checkpoint)

            console.print(checkpoints_table)

    except Exception as e:
        if show_traceback:
            handle_error(e, "showing recovery info")
        else:
            console.print(f"❌ [red]Error showing recovery info: {e}[/red]")
            console.print("💡 [yellow]Use --traceback to see full error details[/yellow]")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("debug")
def debug_execution(
        execution_id: str = typer.Argument(..., help="Execution ID to debug"),
):
    """Debug an execution with full error details and traces."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        console.print(f"🔍 [bold blue]Debugging execution: {execution_id}[/bold blue]\n")

        # Get execution
        execution = repository.get_graph_execution(execution_id)
        if not execution:
            console.print(f"❌ [red]Execution not found: {execution_id}[/red]")
            raise typer.Exit(1)

        # Show basic info
        console.print(f"📊 Graph: {execution.graph_name}")
        console.print(f"📊 Status: {execution.status.value}")
        console.print(f"📊 Started: {execution.started_at}")
        console.print(f"📊 Nodes: {execution.completed_nodes}/{execution.total_nodes} completed")

        if execution.failed_nodes > 0:
            console.print(f"❌ Failed nodes: {execution.failed_nodes}")

        # Show error details
        if execution.error_message:
            console.print("\n🚨 [bold red]Main Error:[/bold red]")
            error_panel = Panel(
                execution.error_message,
                title="Error Message",
                border_style="red"
            )
            console.print(error_panel)

        if execution.error_traceback:
            console.print("\n📋 [bold red]Error Traceback:[/bold red]")
            tb_panel = Panel(
                Syntax(execution.error_traceback, "python", theme="monokai", line_numbers=True),
                title="Full Traceback",
                border_style="red",
                expand=False
            )
            console.print(tb_panel)

        # Show failed nodes
        nodes = repository.get_node_executions_for_graph(execution_id)
        failed_nodes = [n for n in nodes if n.is_failed]

        if failed_nodes:
            console.print(f"\n❌ [bold red]Failed Nodes ({len(failed_nodes)}):[/bold red]")

            for node in failed_nodes:
                console.print(f"\n🔸 [bold yellow]{node.node_name}[/bold yellow] (Run ID: {node.run_id[:8]})")
                console.print(
                    f"   Duration: {node.duration_seconds:.3f}s" if node.duration_seconds else "   Duration: N/A")

                if node.error_message:
                    error_panel = Panel(
                        f"[red]{node.error_type}[/red]: {node.error_message}",
                        title=f"Node Error: {node.node_name}",
                        border_style="red"
                    )
                    console.print(error_panel)

        # Show recovery options
        recovery_info = repository.get_recovery_info(execution_id)
        if recovery_info and recovery_info.can_recover:
            console.print(f"\n🔄 [bold green]Recovery Available![/bold green]")
            console.print(f"   Last checkpoint: {recovery_info.last_successful_checkpoint}")
            console.print(f"   Available checkpoints: {len(recovery_info.available_checkpoints)}")

        # Show database connection info
        console.print(f"\n🗄️  [bold blue]Database Info:[/bold blue]")
        console.print(f"   URL: {settings.database_url}")
        console.print(f"   Tables initialized: ✅")

    except Exception as e:
        handle_error(e, "debugging execution")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("cleanup")
def cleanup_old_executions(
        days: int = typer.Option(30, "--days", help="Keep executions from last N days"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without deleting"),
        confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """Clean up old execution records."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        if not dry_run and not confirm:
            proceed = typer.confirm(f"This will delete execution records older than {days} days. Continue?")
            if not proceed:
                console.print("❌ [yellow]Operation cancelled[/yellow]")
                return

        if dry_run:
            # Count what would be deleted
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            with repository.get_session() as session:
                from .models import GraphExecution
                count = session.query(GraphExecution).filter(
                    GraphExecution.created_at < cutoff_date
                ).count()

            console.print(f"🔍 [yellow]Dry run: Would delete {count} execution records older than {days} days[/yellow]")
        else:
            with console.status("Cleaning up old records..."):
                deleted_count = repository.cleanup_old_executions(days)

            console.print(f"✅ [green]Cleaned up {deleted_count} old execution records[/green]")

    except Exception as e:
        if show_traceback:
            handle_error(e, "cleanup operation")
        else:
            console.print(f"❌ [red]Error during cleanup: {e}[/red]")
            console.print("💡 [yellow]Use --traceback to see full error details[/yellow]")
        raise typer.Exit(1)
    finally:
        try:
            repository.close()
        except:
            pass


@app.command("stats")
def show_statistics(
        days: Optional[int] = typer.Option(7, "--days", help="Show stats for last N days"),
        graph_name: Optional[str] = typer.Option(None, "--graph", help="Filter by graph name"),
        show_traceback: bool = typer.Option(False, "--traceback", help="Show full tracebacks on errors")
):
    """Show execution statistics."""
    try:
        settings = GraphLoggerSettings.create_default()
        repository = GraphLoggerRepository(settings)

        # Calculate date filter
        start_date = None
        if days:
            start_date = datetime.utcnow() - timedelta(days=days)

        with console.status("Calculating statistics..."):
            executions = repository.list_graph_executions(
                graph_name=graph_name,
                start_date=start_date,
                limit=1000  # Get more for stats
            )

        if not executions:
            console.print("🔭 [yellow]No executions found[/yellow]")
            return

        # Calculate statistics
        total_executions = len(executions)
        completed = sum(1 for e in executions if e.is_completed)
        failed = sum(1 for e in executions if e.is_failed)
        running = sum(1 for e in executions if e.is_running)

        total_duration = sum(e.duration_seconds for e in executions if e.duration_seconds)
        avg_duration = total_duration / max(1, len([e for e in executions if e.duration_seconds]))

        total_nodes = sum(e.total_nodes for e in executions)
        total_completed_nodes = sum(e.completed_nodes for e in executions)
        total_failed_nodes = sum(e.failed_nodes for e in executions)

        # Create statistics table
        stats_table = Table(title=f"Execution Statistics (Last {days} days)" if days else "All Time Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Executions", str(total_executions))
        stats_table.add_row("Completed", f"{completed} ({(completed / max(1, total_executions) * 100):.1f}%)")
        stats_table.add_row("Failed", f"{failed} ({(failed / max(1, total_executions) * 100):.1f}%)")
        stats_table.add_row("Running", str(running))
        stats_table.add_row("Average Duration", f"{avg_duration:.2f}s")
        stats_table.add_row("Total Nodes Executed", str(total_nodes))
        stats_table.add_row("Node Success Rate", f"{(total_completed_nodes / max(1, total_nodes) * 100):.1f}%")

        console.print(stats_table)

        # Graph breakdown if not filtered
        if not graph_name:
            graph_stats = {}
            for execution in executions:
                if execution.graph_name not in graph_stats:
                    graph_stats[execution.graph_name] = {
                        'total': 0, 'completed': 0, 'failed': 0, 'running': 0
                    }

                graph_stats[execution.graph_name]['total'] += 1
                if execution.is_completed:
                    graph_stats[execution.graph_name]['completed'] += 1
                elif execution.is_failed:
                    graph_stats[execution.graph_name]['failed'] += 1
                elif execution.is_running:
                    graph_stats[execution.graph_name]['running'] += 1

            if graph_stats:
                console.print("\n")
                graph_table = Table(title="By Graph Type")
                graph_table.add_column("Graph Name", style="blue")
                graph_table.add_column("Total", style="white")
                graph_table.add_column("Completed", style="green")
                graph_table.add_column("Failed", style="red")
                # graph_table.add_column("Running", style="
    finally:
        repository.close()