# examples/fastapi_example.py

"""FastAPI example using LangGraph Logger for tracking graph executions."""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# LangChain and LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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
    title="LangGraph Logger FastAPI Example",
    description="Example showing how to use LangGraph Logger with FastAPI",
    version="1.0.0"
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


# Graph State Definition
class GraphState(BaseModel):
    """State for our example graph."""
    messages: List[Dict[str, Any]] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_step: str = "start"
    processed_count: int = 0


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
            "timestamp": str(asyncio.get_event_loop().time())
        })

        state.current_step = "llm_processed"
        state.processed_count += 1
        return state

    def postprocess_node(state: GraphState) -> GraphState:
        """Postprocess the response."""
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
            "timestamp": str(asyncio.get_event_loop().time())
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

        # Execute with logging
        with logged_graph_execution(
                graph_name="chat_processing_graph",
                initial_state=initial_state.model_dump(),
                database_url=DATABASE_URL,
                tags=request.tags,
                metadata=metadata
        ) as callback:

            # Execute graph
            result = await asyncio.to_thread(
                graph.invoke,
                initial_state,
                config={"callbacks": [callback]}
            )

            # The callback context manager handles finalization automatically
            return GraphExecutionResponse(
                execution_id=callback.execution_id,
                status="completed",
                result=result.model_dump() if hasattr(result, 'model_dump') else result
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
            "timestamp": str(asyncio.get_event_loop().time())
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

        with logged_graph_execution(
                graph_name="chat_processing_graph_async",
                initial_state=initial_state.model_dump(),
                database_url=DATABASE_URL,
                tags=request.tags,
                metadata=metadata
        ) as callback:

            result = await asyncio.to_thread(
                graph.invoke,
                initial_state,
                config={"callbacks": [callback]}
            )

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
                raise HTTPException(status_code=400, f"Invalid status: {status}")

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
                "status": execution.status,
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
        metrics = repository.get_execution_metrics(execution_id)

        return {
            "execution": {
                "id": execution.id,
                "graph_name": execution.graph_name,
                "status": execution.status,
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
                    "status": node.status,
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
        completed = sum(1 for e in executions if e.is_completed)
        failed = sum(1 for e in executions if e.is_failed)
        running = sum(1 for e in executions if e.is_running)

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
    print("🚀 FastAPI LangGraph Logger Example started")
    print(f"📊 Database: {DATABASE_URL}")
    print("📝 Available endpoints:")
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
        "fastapi_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )