# examples/fastapi_example_fixed.py

"""Fixed FastAPI example using LangGraph Logger for tracking graph executions."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough

# LangGraph Logger imports
from langgraph_logger import (
    GraphExecutionCallback,
    GraphLoggerSettings,
    GraphLoggerRepository,
    logged_graph_execution
)

# FastAPI app setup
app = FastAPI(
    title="LangGraph Logger FastAPI Example (Fixed)",
    description="Fixed example showing how to use LangGraph Logger with FastAPI",
    version="1.1.0"
)

# Configuration
DATABASE_URL = "sqlite:///./fastapi_graph_executions.db"
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual API key

# Initialize logger settings
logger_settings = GraphLoggerSettings(
    database_url=DATABASE_URL,
    enable_rich_output=False,  # Disable rich output for FastAPI
    enable_console_logging=True,
    auto_save_state=True,
    recovery_checkpoint_interval=3
)

# Global repository for API endpoints
repository = GraphLoggerRepository(logger_settings)

# Initialize database tables
repository.create_tables()


# Pydantic models for API
class GraphExecutionRequest(BaseModel):
    """Request model for graph execution."""
    initial_message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class GraphExecutionResponse(BaseModel):
    """Response model for graph execution."""
    execution_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExecutionListResponse(BaseModel):
    """Response model for listing executions."""
    executions: List[Dict[str, Any]]
    total: int


# Graph State Definition - Fixed to use dict instead of BaseModel
class GraphState(dict):
    """State for our example graph - using dict for proper serialization."""

    def __init__(self, **kwargs):
        super().__init__()
        self.update({
            'messages': kwargs.get('messages', []),
            'user_id': kwargs.get('user_id'),
            'session_id': kwargs.get('session_id'),
            'current_step': kwargs.get('current_step', 'start'),
            'processed_count': kwargs.get('processed_count', 0)
        })

    @property
    def messages(self):
        return self.get('messages', [])

    @messages.setter
    def messages(self, value):
        self['messages'] = value

    @property
    def current_step(self):
        return self.get('current_step', 'start')

    @current_step.setter
    def current_step(self, value):
        self['current_step'] = value

    @property
    def processed_count(self):
        return self.get('processed_count', 0)

    @processed_count.setter
    def processed_count(self, value):
        self['processed_count'] = value


def create_chat_graph() -> StateGraph:
    """Create a simple chat processing graph."""

    # Initialize LLM (you'll need to set your OpenAI API key)
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )
    except Exception:
        # Fallback for demo purposes - replace with actual LLM
        llm = None

    def preprocess_node(state: GraphState) -> GraphState:
        """Preprocess the input message."""
        if not isinstance(state, GraphState):
            state = GraphState(**state)
        if state.messages:
            last_message = state.messages[-1]
            # Add preprocessing metadata
            last_message["preprocessed"] = True
            last_message["length"] = len(last_message.get("content", ""))

        state.current_step = "preprocessing_done"
        state.processed_count += 1
        return state

    def llm_node(state: GraphState) -> GraphState:
        """Process message with LLM."""
        if not isinstance(state, GraphState):
            state = GraphState(**state)
        if not llm:
            # Mock response for demo
            response_content = f"Mock response to: {state.messages[-1].get('content', '')}"
        else:
            try:
                # Convert to LangChain messages
                lc_messages = []
                for msg in state.messages:
                    if msg.get("role") == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))

                # Get LLM response
                response = llm.invoke(lc_messages)
                response_content = response.content
            except Exception as e:
                response_content = f"Error processing with LLM: {str(e)}"

        # Add AI response to messages
        state.messages.append({
            "role": "assistant",
            "content": response_content,
            "timestamp": str(asyncio.get_event_loop().time()) if asyncio._get_running_loop() else str(uuid.uuid4())
        })

        state.current_step = "llm_processed"
        state.processed_count += 1
        return state

    def postprocess_node(state: GraphState) -> GraphState:
        """Postprocess the response."""
        if not isinstance(state, GraphState):
            state = GraphState(**state)
        if state.messages:
            last_message = state.messages[-1]
            # Add postprocessing metadata
            last_message["postprocessed"] = True
            last_message["word_count"] = len(last_message.get("content", "").split())

        state.current_step = "completed"
        state.processed_count += 1
        return state

    # Build the graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("llm", llm_node)
    graph.add_node("postprocess", postprocess_node)

    # Add edges
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "llm")
    graph.add_edge("llm", "postprocess")
    graph.add_edge("postprocess", END)

    return graph.compile()


# Custom callback class to handle the serialization issues
class FixedGraphExecutionCallback(GraphExecutionCallback):
    """Fixed callback that handles UUID and state serialization properly."""

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a node/chain starts execution - fixed version."""
        self._ensure_execution_started()

        # Fix run_id conversion
        run_id = str(kwargs.get("run_id", f"unknown_run_{uuid.uuid4()}"))
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

        # Convert inputs to dict if it's a GraphState
        if hasattr(inputs, '__dict__'):
            input_dict = dict(inputs) if isinstance(inputs, dict) else inputs.__dict__
        else:
            input_dict = inputs if isinstance(inputs, dict) else {"state": str(inputs)}

        # Log to database
        if self.execution_id and not self.execution_id.startswith("local_"):
            try:
                from langgraph_logger.dto import NodeExecutionCreate
                node_data = NodeExecutionCreate(
                    execution_id=self.execution_id,
                    node_name=node_name,
                    run_id=run_id,  # Already converted to string
                    input_data=input_dict if len(str(input_dict)) < 10000 else {"_truncated": True},
                    extra_metadata=kwargs.get("extra_metadata")
                )
                self.repository.create_node_execution(node_data)
            except Exception as e:
                print(f"Failed to create node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_start(node_name, input_dict, run_id)

        # Update current state with inputs (simple merge)
        if input_dict and self.settings.auto_save_state:
            self.current_state.update(input_dict)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a node/chain ends successfully - fixed version."""
        import time

        run_id = str(kwargs.get("run_id", "unknown_run"))

        with self._lock:
            if run_id not in self.node_start_times or run_id not in self.node_names:
                print(f"Node end called for unknown run_id: {run_id}")
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

        # Convert outputs to dict if needed
        if hasattr(outputs, '__dict__'):
            output_dict = dict(outputs) if isinstance(outputs, dict) else outputs.__dict__
        else:
            output_dict = outputs if isinstance(outputs, dict) else {"state": str(outputs)}

        # Update database
        if self.execution_id and not self.execution_id.startswith("local_"):
            try:
                from langgraph_logger.dto import NodeExecutionUpdate, NodeStatus
                update_data = NodeExecutionUpdate(
                    status=NodeStatus.COMPLETED,
                    output_data=output_dict if len(str(output_dict)) < 10000 else {"_truncated": True}
                )
                self.repository.update_node_execution(run_id, update_data)
                self.repository.update_execution_stats(self.execution_id)
            except Exception as e:
                print(f"Failed to update node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_end(node_name, output_dict, execution_time, run_id)

        # Update current state with outputs
        if output_dict and self.settings.auto_save_state:
            self.current_state.update(output_dict)

        # Create checkpoint if needed
        self._maybe_create_checkpoint()

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a node/chain encounters an error - fixed version."""
        import time

        run_id = str(kwargs.get("run_id", "unknown_run"))

        with self._lock:
            if run_id not in self.node_start_times or run_id not in self.node_names:
                print(f"Node error called for unknown run_id: {run_id}")
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
                from langgraph_logger.dto import NodeExecutionUpdate, NodeStatus
                update_data = NodeExecutionUpdate(
                    status=NodeStatus.FAILED,
                    error_message=str(error),
                    error_type=type(error).__name__
                )
                self.repository.update_node_execution(run_id, update_data)
                self.repository.update_execution_stats(self.execution_id)
            except Exception as e:
                print(f"Failed to update node execution record: {e}")

        # Console logging
        if self.settings.enable_console_logging:
            self._log_node_error(node_name, error, execution_time, run_id)

        # Create error checkpoint
        self._create_error_checkpoint(node_name, error)


def run_graph_in_thread(graph, initial_state, callback):
    """Run graph in a thread with proper event loop handling."""
    import threading
    import asyncio

    result = None
    error = None

    def thread_target():
        nonlocal result, error
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Execute graph
            result = graph.invoke(initial_state, config={"callbacks": [callback]})
        except Exception as e:
            error = e
        finally:
            # Clean up event loop
            try:
                loop.close()
            except:
                pass

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()

    if error:
        raise error

    return result


# FastAPI endpoints

@app.post("/execute", response_model=GraphExecutionResponse)
async def execute_graph(
        request: GraphExecutionRequest,
        background_tasks: BackgroundTasks
) -> GraphExecutionResponse:
    """Execute a graph with logging."""

    # Generate execution ID
    execution_id = str(uuid.uuid4())

    # Prepare initial state
    initial_state = GraphState(
        messages=[{
            "role": "user",
            "content": request.initial_message,
            "timestamp": str(uuid.uuid4())  # Use UUID instead of event loop time
        }],
        user_id=request.user_id,
        session_id=request.session_id,
        current_step="start"
    )

    # Prepare metadata
    metadata = request.extra_metadata or {}
    metadata.update({
        "user_id": request.user_id,
        "session_id": request.session_id,
        "api_endpoint": "/execute",
        "initial_message_length": len(request.initial_message)
    })

    try:
        # Create graph
        graph = create_chat_graph()

        # Create callback with fixed implementation
        callback = FixedGraphExecutionCallback(
            graph_name="chat_processing_graph",
            settings=logger_settings,
            repository=repository,
            initial_state=dict(initial_state),
            tags=request.tags,
            extra_metadata=metadata
        )

        # Execute with proper thread handling
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_graph_in_thread,
            graph,
            initial_state,
            callback
        )

        # Finalize callback
        callback.finalize_execution(final_state=dict(result) if result else None)

        return GraphExecutionResponse(
            execution_id=callback.execution_id,
            status="completed",
            result=dict(result) if result else None
        )

    except Exception as e:
        return GraphExecutionResponse(
            execution_id=execution_id,
            status="failed",
            error=str(e)
        )


@app.post("/execute-async")
async def execute_graph_async(
        request: GraphExecutionRequest,
        background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start graph execution in background."""

    execution_id = str(uuid.uuid4())

    # Add background task
    background_tasks.add_task(
        execute_graph_background,
        execution_id,
        request
    )

    return {
        "execution_id": execution_id,
        "status": "started",
        "message": "Execution started in background"
    }


async def execute_graph_background(execution_id: str, request: GraphExecutionRequest):
    """Background task for graph execution."""

    initial_state = GraphState(
        messages=[{
            "role": "user",
            "content": request.initial_message,
            "timestamp": str(uuid.uuid4())
        }],
        user_id=request.user_id,
        session_id=request.session_id
    )

    metadata = request.extra_metadata or {}
    metadata.update({
        "user_id": request.user_id,
        "session_id": request.session_id,
        "execution_type": "background"
    })

    try:
        graph = create_chat_graph()

        callback = FixedGraphExecutionCallback(
            graph_name="chat_processing_graph_async",
            settings=logger_settings,
            repository=repository,
            initial_state=dict(initial_state),
            tags=request.tags,
            extra_metadata=metadata
        )

        # Execute in thread
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            run_graph_in_thread,
            graph,
            initial_state,
            callback
        )

        callback.finalize_execution(final_state=dict(result) if result else None)

    except Exception as e:
        print(f"Background execution failed: {e}")


@app.get("/executions", response_model=ExecutionListResponse)
async def list_executions(
        limit: int = 10,
        offset: int = 0,
        graph_name: Optional[str] = None,
        status: Optional[str] = None
) -> ExecutionListResponse:
    """List graph executions."""

    try:
        # Convert status string to enum if provided
        status_enum = None
        if status:
            from langgraph_logger.dto import ExecutionStatus
            try:
                status_enum = ExecutionStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        executions = repository.list_graph_executions(
            graph_name=graph_name,
            status=status_enum,
            limit=limit,
            offset=offset
        )

        # Convert to dict format
        execution_dicts = []
        for execution in executions:
            execution_dicts.append({
                "id": execution.id,
                "graph_name": execution.graph_name,
                "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "total_nodes": execution.total_nodes,
                "completed_nodes": execution.completed_nodes,
                "failed_nodes": execution.failed_nodes,
                "success_rate": execution.success_rate,
                "tags": execution.tags
            })

        return ExecutionListResponse(
            executions=execution_dicts,
            total=len(execution_dicts)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions/{execution_id}")
async def get_execution(execution_id: str) -> Dict[str, Any]:
    """Get detailed execution information."""

    try:
        execution = repository.get_graph_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        # Get additional details
        nodes = repository.get_node_executions_for_graph(execution_id)
        try:
            metrics = repository.get_execution_metrics(execution_id)
        except:
            metrics = None

        return {
            "execution": {
                "id": execution.id,
                "graph_name": execution.graph_name,
                "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": execution.duration_seconds,
                "initial_state": execution.initial_state,
                "final_state": execution.final_state,
                "total_nodes": execution.total_nodes,
                "completed_nodes": execution.completed_nodes,
                "failed_nodes": execution.failed_nodes,
                "success_rate": execution.success_rate,
                "max_parallel_nodes": execution.max_parallel_nodes,
                "extra_metadata": execution.extra_metadata,
                "tags": execution.tags,
                "error_message": execution.error_message
            },
            "nodes": [
                {
                    "id": node.id,
                    "node_name": node.node_name,
                    "run_id": node.run_id,
                    "status": node.status.value if hasattr(node.status, 'value') else str(node.status),
                    "started_at": node.started_at.isoformat(),
                    "completed_at": node.completed_at.isoformat() if node.completed_at else None,
                    "duration_seconds": node.duration_seconds,
                    "error_message": node.error_message
                }
                for node in nodes
            ],
            "metrics": metrics.model_dump() if metrics else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions/{execution_id}/recovery")
async def get_recovery_info(execution_id: str) -> Dict[str, Any]:
    """Get recovery information for a failed execution."""

    try:
        recovery_info = repository.get_recovery_info(execution_id)
        if not recovery_info:
            raise HTTPException(status_code=404, detail="Recovery info not found")

        return recovery_info.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/executions/{execution_id}")
async def delete_execution(execution_id: str) -> Dict[str, str]:
    """Delete an execution record."""

    try:
        execution = repository.get_graph_execution(execution_id)
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        # Note: This would require implementing a delete method in the repository
        # For now, we'll return a placeholder response
        return {"message": "Delete functionality not implemented yet"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics(days: Optional[int] = 7) -> Dict[str, Any]:
    """Get execution statistics."""

    try:
        from datetime import datetime, timedelta

        start_date = None
        if days:
            start_date = datetime.utcnow() - timedelta(days=days)

        executions = repository.list_graph_executions(
            start_date=start_date,
            limit=1000
        )

        if not executions:
            return {"message": "No executions found"}

        # Calculate statistics
        total_executions = len(executions)
        completed = sum(1 for e in executions if hasattr(e, 'is_completed') and e.is_completed)
        failed = sum(1 for e in executions if hasattr(e, 'is_failed') and e.is_failed)
        running = sum(1 for e in executions if hasattr(e, 'is_running') and e.is_running)

        total_duration = sum(e.duration_seconds for e in executions if e.duration_seconds)
        avg_duration = total_duration / max(1, len([e for e in executions if e.duration_seconds]))

        return {
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": (completed / max(1, total_executions)) * 100,
            "average_duration_seconds": avg_duration,
            "total_duration_seconds": total_duration
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("🚀 FastAPI LangGraph Logger Example started (Fixed Version)")
    print(f"📊 Database: {DATABASE_URL}")
    print("🔍 Available endpoints:")
    print("  POST /execute - Execute graph synchronously")
    print("  POST /execute-async - Execute graph in background")
    print("  GET /executions - List executions")
    print("  GET /executions/{id} - Get execution details")
    print("  GET /executions/{id}/recovery - Get recovery info")
    print("  GET /stats - Get statistics")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    repository.close()
    print("👋 FastAPI LangGraph Logger Example shutting down")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "fastapi_example_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )