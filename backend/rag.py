# backend/rag.py
import logging
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import config
from history import get_session_history

logger = logging.getLogger(__name__)

contextualize_q_system_prompt = """
給定一段對話歷史和使用者的最新提問
(該提問可能引用了上文內容)，
請將其改寫為一個獨立的、可被理解的問題。
不需要回答問題，只要重寫它，如果不需要重寫則保持原樣。
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = """
你是{MY_NAME}。請根據以下的上下文片段來回答面試官的問題。
如果你不知道答案，就說不知道，不要編造內容。
請保持專業且自信的語氣。

上下文: {context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
).partial(MY_NAME=config.MY_NAME)

vector_store = PineconeVectorStore(
    index_name=config.PINECONE_INDEX_NAME,
    embedding=OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY, model=config.EMBEDDING_MODEL
    ),
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

history_aware_model = init_chat_model(config.QA_CHAT_MODEL, api_key=config.OPENAI_API_KEY)
history_aware_retriever = create_history_aware_retriever(
    history_aware_model, retriever, contextualize_q_prompt
)

qa_assistant_llm = init_chat_model(config.QA_CHAT_MODEL, api_key=config.OPENAI_API_KEY)
question_answer_chain = create_stuff_documents_chain(qa_assistant_llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)