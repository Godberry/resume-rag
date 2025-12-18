"""LangGraph state schema definition."""

from langchain_core.messages import BaseMessage
from langgraph.graph.state import StateMessage


class GraphState(StateMessage):
    """LangGraph state schema with message history."""
