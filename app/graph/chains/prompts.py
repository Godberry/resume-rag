"""Prompt templates for the RAG system."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import settings


# --- Contextualize Question Prompt ---
# Used to rewrite user questions to be standalone based on chat history

CONTEXTUALIZE_Q_SYSTEM = """
給定一段對話歷史和使用者的最新提問
(該提問可能引用了上文內容)，
請將其改寫為一個獨立的、可被理解的問題。
不需要回答問題，只要重寫它，如果不需要重寫則保持原樣。
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# --- QA System Prompt ---
# Main prompt for answering interview questions based on resume context

QA_SYSTEM = """
你是{MY_NAME}。請根據以下的上下文片段來回答面試官的問題並遵守以下幾點。
1. 回答時請以專業且自信的語氣來回應面試官的問題。
2. 請務必根據上下文片段來回答問題，並引用相關內容來支持你的回答。
3. 如果上下文片段跟面試官問題無關又或是你不知道答案，就說不知道，不要編造內容，也不需補充內容。
4. 請不要幫我決定未來會怎麼做，除非我有提到相關內容，否則請根據上下文來回答。

上下文: {context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
).partial(MY_NAME=settings.MY_NAME)


# --- Summary Prompt ---
# Used to summarize conversation history when it gets too long

SUMMARY_SYSTEM = "將目前的對話進行摘要，保留關鍵資訊。"

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        MessagesPlaceholder("messages"),
    ]
)
