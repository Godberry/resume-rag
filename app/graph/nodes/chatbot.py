"""Chatbot node - main RAG logic."""

import logging
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.core.state import GraphState
from app.services.llm_factory import get_llm
from app.services.retriever import get_retriever
from app.graph.chains.prompts import contextualize_q_prompt, qa_prompt

logger = logging.getLogger(__name__)


def _build_rag_chain():
    """Build the RAG chain with history-aware retriever."""
    llm = get_llm()
    retriever = get_retriever()

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Build chain at module load time
_rag_chain = None


def _get_rag_chain():
    """Lazy initialization of RAG chain."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = _build_rag_chain()
    return _rag_chain


def chatbot(state: GraphState, config: RunnableConfig) -> dict:
    """Process user message and generate RAG response.

    Args:
        state: Current graph state with messages
        config: Runnable configuration

    Returns:
        Dict with new AI message to add to state
    """
    messages = state["messages"]
    last_message = messages[-1]
    history = messages[:-1]

    logger.info("Chatbot node processing message: %s", last_message.content[:50])

    rag_chain = _get_rag_chain()
    response = rag_chain.invoke(
        {"input": last_message.content, "chat_history": history},
        config=config,
    )

    logger.info("Chatbot node generated response")
    return {"messages": [AIMessage(content=response["answer"])]}
