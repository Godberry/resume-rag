"""Summarizer node - condenses conversation history when too long."""

import logging
from typing import Literal
from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from app.core.state import GraphState
from app.config import settings
from app.services.llm_factory import get_llm
from app.graph.chains.prompts import summary_prompt

logger = logging.getLogger(__name__)


def summarizer(state: GraphState, config: RunnableConfig) -> dict:
    """Summarize conversation history and replace with summary message.

    This node is called conditionally when message count exceeds threshold.
    It creates a summary of all messages and returns:
    - A new SystemMessage containing the summary
    - RemoveMessage commands to delete old messages

    Args:
        state: Current graph state with messages
        config: Runnable configuration

    Returns:
        Dict with summary message and delete commands
    """
    messages = state["messages"]
    logger.info("Summarizer node triggered with %d messages", len(messages))

    # Generate summary
    llm = get_llm()
    chain = summary_prompt | llm
    response = chain.invoke({"messages": messages}, config=config)

    # Create deletion commands for all existing messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages if m.id]

    # Create new summary system message
    summary_content = f"先前對話摘要: {response.content}"
    system_message = SystemMessage(content=summary_content)

    logger.info("Summarizer created summary, removing %d messages", len(delete_messages))
    return {"messages": [system_message] + delete_messages}


def should_summarize(state: GraphState) -> Literal["summarizer", "__end__"]:
    """Determine if summarization is needed based on message count.

    Args:
        state: Current graph state with messages

    Returns:
        "summarizer" if messages exceed threshold, END otherwise
    """
    messages = state["messages"]
    if len(messages) > settings.RECENT_MESSAGES_TO_KEEP:
        logger.info(
            "Message count %d exceeds threshold %d, will summarize",
            len(messages),
            settings.RECENT_MESSAGES_TO_KEEP,
        )
        return "summarizer"
    return END
