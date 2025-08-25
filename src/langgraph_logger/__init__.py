# src/langgraph_logger/__init__.py

"""
LangGraph Logger: Advanced logging and state management for LangGraph executions.

This package provides comprehensive logging, monitoring, and state management
capabilities for LangGraph workflows with database persistence.
"""

from .callback import GraphExecutionCallback
from .models import GraphExecution, NodeExecution, ExecutionState
from .repository import GraphLoggerRepository
from .settings import GraphLoggerSettings
from .utils import logged_graph_execution


__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "GraphExecutionCallback",
    "GraphExecution",
    "NodeExecution",
    "ExecutionState",
    "GraphLoggerRepository",
    "GraphLoggerSettings",
    "logged_graph_execution"
]