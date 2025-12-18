"""LangGraph workflow definition - assembles the state graph."""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.state import GraphState
from app.graph.nodes.chatbot import chatbot
from app.graph.nodes.summarizer import summarizer, should_summarize

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build the conversation graph.

    Graph flow:
        START -> chatbot -> [should_summarize] -> summarizer -> END
                                              |-> END

    Returns:
        Compiled StateGraph with checkpointer
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("summarizer", summarizer)

    # Add edges
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges(
        "chatbot",
        should_summarize,
        {
            "summarizer": "summarizer",
            END: END,
        },
    )
    workflow.add_edge("summarizer", END)

    return workflow


# Build and compile the graph with MemorySaver checkpointer
_checkpointer = MemorySaver()
_workflow = build_graph()
graph_app = _workflow.compile(checkpointer=_checkpointer)

logger.info("LangGraph workflow compiled successfully")
