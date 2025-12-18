# backend/rag.py
import logging
from typing import TypedDict, Annotated, List, Literal

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, RemoveMessage, trim_messages, AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import config

logger = logging.getLogger(__name__)

# --- Prompts & LLM Setup ---

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
你是{MY_NAME}。請根據以下的上下文片段來回答面試官的問題並遵守以下幾點。
1. 回答時請以專業且自信的語氣來回應面試官的問題。
2. 請務必根據上下文片段來回答問題，並引用相關內容來支持你的回答。
3. 如果上下文片段跟面試官問題無關又或是你不知道答案，就說不知道，不要編造內容，也不需補充內容。
4. 請不要幫我決定未來會怎麼做，除非我有提到相關內容，否則請根據上下文來回答。

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

# Initialize Models
llm = init_chat_model(config.QA_CHAT_MODEL, api_key=config.OPENAI_API_KEY)

# --- Chains ---
# We keep the chain logic as it's a good way to encapsulate retrieval + generation
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- LangGraph Definition ---

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chatbot(state: GraphState, config: RunnableConfig):
    messages = state["messages"]
    # Last message is the user input
    last_message = messages[-1]
    # Previous messages are history
    history = messages[:-1]
    
    # Invoke RAG chain
    response = rag_chain.invoke(
        {"input": last_message.content, "chat_history": history},
        config=config
    )
    
    return {"messages": [AIMessage(content=response["answer"])]}

def summarizer(state: GraphState, config: RunnableConfig):
    messages = state["messages"]
    # Logic to summarize if too long
    # But wait, this node is only called IF we decide to summarize.
    # So we just do the summarization here.
    
    # Create a summary using the LLM
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "將目前的對話進行摘要，保留關鍵資訊。"),
            MessagesPlaceholder("messages"),
        ]
    )
    chain = summary_prompt | llm
    response = chain.invoke({"messages": messages})
    
    # Create deletion messages for all but the last 2 messages (usually summary + new user msg? 
    # Or strict logic: summarize ALL previous, remove them, add Summary msg at start).
    # Simple strategy: Summary replaces everything except the very last exchange? 
    # Or just keep the summary and remove everything?
    
    # User requirement: "Return RemoveMessage to delete old messages, and insert new Summary SystemMessage".
    # We will remove all messages except the most recent ones if we want to keep context flowing, 
    # but strictly speaking, a summary usually replaces the history.
    # Let's remove all messages that were summarized.
    
    delete_messages = [RemoveMessage(id=m.id) for m in messages if m.id] 
    # Note: messages need IDs for RemoveMessage to work well. Add_messages assigns IDs usually.
    
    # Since we might not have IDs if we didn't persistence load them... 
    # But LangGraph assigns IDs.
    
    system_message = SystemMessage(content=f"先前對話摘要: {response.content}")
    
    return {"messages": [system_message] + delete_messages}

def should_summarize(state: GraphState) -> Literal["summarizer", END]:
    messages = state["messages"]
    # Check if length > threshold
    if len(messages) > config.RECENT_MESSAGES_TO_KEEP:
        return "summarizer"
    return END

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("chatbot", chatbot)
workflow.add_node("summarizer", summarizer)

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    should_summarize,
    {
        "summarizer": "summarizer",
        END: END
    }
)
workflow.add_edge("summarizer", END)

# Compile
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)