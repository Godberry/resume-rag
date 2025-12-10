# backend/history.py
import logging
from typing import List
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import tiktoken

import config

logger = logging.getLogger(__name__)

FIRESTORE_CLIENT = firestore.Client(
    project=config.FIRESTORE_PROJECT_ID,
    database=config.FIRESTORE_DATABASE,
)

try:
    _encoding = tiktoken.encoding_for_model(config.TOKEN_ENCODING_MODEL)
except Exception:
    _encoding = tiktoken.get_encoding("cl100k_base")

summary_system_prompt = """
你是一個對話整理助手。以下是一段與面試官的歷史對話，
請將它濃縮成一小段摘要，保留關於我（受面試者）的背景、經驗、專案與需求等重要資訊，
之後會用這個摘要幫助回答後續問題。請用自然的中文第一人稱敘述。
"""
summary_prompt = ChatPromptTemplate.from_messages(
    [("system", summary_system_prompt), ("human", "{history_text}")]
)

summary_llm = init_chat_model("gpt-4.1-mini", api_key=config.OPENAI_API_KEY)


def get_session_history(session_id: str) -> FirestoreChatMessageHistory:
    if not session_id:
        raise ValueError("session_id 不可為空")
    logger.info("get_session_history start, session_id=%s", session_id)
    history = FirestoreChatMessageHistory(
        session_id=session_id,
        collection=config.FIRESTORE_COLLECTION,
        client=FIRESTORE_CLIENT,
    )
    logger.info("get_session_history done")
    return history


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding.encode(text))


def _estimate_history_tokens(messages: List) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        total += _count_tokens(content if isinstance(content, str) else str(content))
    return total


def summarize_history_if_too_long(session_id: str) -> None:
    history = get_session_history(session_id)
    messages = history.messages

    total_tokens = _estimate_history_tokens(messages)
    logger.info(
        "Session %s history estimated tokens: %d (limit %d)",
        session_id,
        total_tokens,
        config.MAX_HISTORY_TOKENS,
    )
    if total_tokens <= config.MAX_HISTORY_TOKENS:
        return

    if len(messages) <= config.RECENT_MESSAGES_TO_KEEP:
        old_messages = messages
        recent_messages = []
    else:
        old_messages = messages[:-config.RECENT_MESSAGES_TO_KEEP]
        recent_messages = messages[-config.RECENT_MESSAGES_TO_KEEP :]

    if not old_messages:
        return

    lines = []
    for m in old_messages:
        role = getattr(m, "type", "unknown").upper()
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        lines.append(f"{role}: {content}")
    history_text = "\n".join(lines)
    if not history_text.strip():
        return

    summary_chain = summary_prompt | summary_llm
    summary_msg = summary_chain.invoke({"history_text": history_text})
    summary_text = getattr(summary_msg, "content", "")

    summary_system_msg = SystemMessage(
        content=f"以下是更早期對話的摘要，供後續回答問題時參考：\n{summary_text}"
    )

    history.clear()
    history.add_message(summary_system_msg)
    for msg in recent_messages:
        history.add_message(msg)

    logger.info(
        "History for session %s rewritten to %d messages (1 summary + %d recent).",
        session_id,
        1 + len(recent_messages),
        len(recent_messages),
    )