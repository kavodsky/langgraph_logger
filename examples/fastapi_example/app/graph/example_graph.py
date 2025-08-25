# app/graph/example_graph.py
"""Example LangGraph implementation for demonstration."""

import time
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """State model for the example graph."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: str = Field(default="start")
    processed_count: int = Field(default=0)
    result: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # def keys(self):
    #     return ["messages", "current_step", "processed_count", "result", "metadata"]
    #
    # def __getitem__(self, key):
    #     return self.model_dump().get(key)

def input_processor_node(state: GraphState) -> Dict[str, Any]:
    """Process input and prepare for analysis."""
    print(f"Processing input in state: {state.current_step}")

    # Simulate some processing time
    time.sleep(0.5)

    # Add a message to track processing
    state.messages.append({
        "role": "system",
        "content": "Input processed and validated",
        "timestamp": time.time()
    })

    # Update metadata
    state.metadata["input_processed_at"] = time.time()
    state.metadata["input_size"] = len(str(state.messages))

    return {
        "current_step": "analysis",
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "metadata": state.metadata
    }


def analysis_node(state: GraphState) -> Dict[str, Any]:
    """Perform analysis on the processed input."""
    print(f"Performing analysis in state: {state.current_step}")

    # Simulate analysis work
    time.sleep(1.0)

    # Generate some analysis results
    analysis_result = {
        "sentiment": "positive",
        "confidence": 0.85,
        "key_topics": ["example", "demonstration", "testing"],
        "word_count": len(state.messages) * 10  # Mock word count
    }

    state.messages.append({
        "role": "assistant",
        "content": f"Analysis completed: {analysis_result['sentiment']} sentiment detected",
        "timestamp": time.time()
    })

    state.result.update(analysis_result)
    state.metadata["analysis_completed_at"] = time.time()

    return {
        "current_step": "decision",
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": state.result,
        "metadata": state.metadata
    }


def decision_node(state: GraphState) -> Dict[str, Any]:
    """Make decisions based on analysis results."""
    print(f"Making decision in state: {state.current_step}")

    time.sleep(0.3)

    # Decision logic based on analysis
    confidence = state.result.get("confidence", 0.0)

    if confidence > 0.8:
        next_step = "enhancement"
        decision = "high_confidence_path"
    elif confidence > 0.5:
        next_step = "validation"
        decision = "medium_confidence_path"
    else:
        next_step = "error_handling"
        decision = "low_confidence_path"

    state.messages.append({
        "role": "system",
        "content": f"Decision made: {decision} (confidence: {confidence})",
        "timestamp": time.time()
    })

    state.result["decision"] = decision
    state.result["next_action"] = next_step
    state.metadata["decision_made_at"] = time.time()

    return {
        "current_step": next_step,
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": state.result,
        "metadata": state.metadata
    }


def enhancement_node(state: GraphState) -> Dict[str, Any]:
    """Enhance results for high confidence cases."""
    print(f"Enhancing results in state: {state.current_step}")

    time.sleep(0.7)

    # Add enhancements
    enhancements = {
        "detailed_analysis": True,
        "recommendations": [
            "Continue with current approach",
            "Consider additional data sources",
            "Implement monitoring"
        ],
        "quality_score": 0.95
    }

    state.result.update(enhancements)
    state.messages.append({
        "role": "assistant",
        "content": "Results enhanced with additional insights",
        "timestamp": time.time()
    })

    return {
        "current_step": "finalization",
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": state.result,
        "metadata": state.metadata
    }


def validation_node(state: GraphState) -> Dict[str, Any]:
    """Validate results for medium confidence cases."""
    print(f"Validating results in state: {state.current_step}")

    time.sleep(0.8)

    # Simulate validation
    validation_passed = True  # Mock validation

    if validation_passed:
        state.result["validation_status"] = "passed"
        state.result["validated_confidence"] = state.result.get("confidence", 0.0) + 0.1
        next_step = "finalization"
    else:
        state.result["validation_status"] = "failed"
        next_step = "error_handling"

    state.messages.append({
        "role": "system",
        "content": f"Validation {state.result['validation_status']}",
        "timestamp": time.time()
    })

    return {
        "current_step": next_step,
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": state.result,
        "metadata": state.metadata
    }


def error_handling_node(state: GraphState) -> Dict[str, Any]:
    """Handle low confidence or error cases."""
    print(f"Handling errors in state: {state.current_step}")

    time.sleep(0.4)

    # Generate error recovery
    state.result["error_handled"] = True
    state.result["fallback_result"] = {
        "status": "recovered",
        "message": "Applied fallback processing",
        "confidence": 0.3
    }

    state.messages.append({
        "role": "system",
        "content": "Error handled with fallback processing",
        "timestamp": time.time()
    })

    return {
        "current_step": "finalization",
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": state.result,
        "metadata": state.metadata
    }


def finalization_node(state: GraphState) -> Dict[str, Any]:
    """Finalize processing and prepare output."""
    print(f"Finalizing in state: {state.current_step}")

    time.sleep(0.2)

    # Finalize results
    final_result = {
        "status": "completed",
        "total_steps": state.processed_count + 1,
        "processing_time": time.time() - state.metadata.get("input_processed_at", time.time()),
        "summary": state.result
    }

    state.messages.append({
        "role": "assistant",
        "content": "Processing completed successfully",
        "timestamp": time.time()
    })

    state.metadata["completed_at"] = time.time()

    return {
        "current_step": "completed",
        "processed_count": state.processed_count + 1,
        "messages": state.messages,
        "result": final_result,
        "metadata": state.metadata
    }


def should_continue(state: GraphState) -> str:
    """Conditional logic for routing between nodes."""
    current_step = state.current_step

    if current_step == "start":
        return "input_processor"
    elif current_step == "analysis":
        return "decision"
    elif current_step == "decision":
        next_action = state.result.get("next_action", "finalization")
        return next_action
    elif current_step in ["enhancement", "validation", "error_handling"]:
        return "finalization"
    else:
        return "end"


def create_example_graph() -> StateGraph:
    """Create and configure the example graph."""

    # Create the state graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("input_processor", input_processor_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("decision", decision_node)
    graph.add_node("enhancement", enhancement_node)
    graph.add_node("validation", validation_node)
    graph.add_node("error_handling", error_handling_node)
    graph.add_node("finalization", finalization_node)

    # Set entry point
    graph.set_entry_point("input_processor")

    # Add conditional edges
    graph.add_conditional_edges(
        "input_processor",
        should_continue,
        {
            "analysis": "analysis",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "analysis",
        should_continue,
        {
            "decision": "decision",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "decision",
        should_continue,
        {
            "enhancement": "enhancement",
            "validation": "validation",
            "error_handling": "error_handling",
            "finalization": "finalization",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "enhancement",
        should_continue,
        {
            "finalization": "finalization",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "validation",
        should_continue,
        {
            "finalization": "finalization",
            "error_handling": "error_handling",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "error_handling",
        should_continue,
        {
            "finalization": "finalization",
            "end": "__end__"
        }
    )

    graph.add_conditional_edges(
        "finalization",
        should_continue,
        {
            "end": "__end__"
        }
    )

    return graph.compile()