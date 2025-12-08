"""FastAPI backend for resume RAG chatbot."""
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-rag")
FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "(default)")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "chat_sessions")
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))  # 歷史 token 上限
RECENT_MESSAGES_TO_KEEP = int(os.getenv("RECENT_MESSAGES_TO_KEEP", "6"))  # 保留最近幾則原始訊息
TOKEN_ENCODING_MODEL = os.getenv("TOKEN_ENCODING_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 環境變數未設定。")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY 環境變數未設定。")
if not FIRESTORE_PROJECT_ID:
    raise RuntimeError("FIRESTORE_PROJECT_ID 環境變數未設定。")

MY_NAME = "許皓翔"

try:
    _encoding = tiktoken.encoding_for_model(TOKEN_ENCODING_MODEL)
except Exception:
    _encoding = tiktoken.get_encoding("cl100k_base")

# 初始化 FastAPI
app = FastAPI(title="Resume RAG API",
              description="基於履歷的 RAG 聊天後端",
              version="1.0.0")

# CORS（前端本地開發預設允許）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 添加 session_id

class ChatResponse(BaseModel):
    answer: str
    # 未來如果要回傳來源片段，可在這裡擴充
    # sources: List[str] = []

contextualize_q_system_prompt = """
給定一段對話歷史和使用者的最新提問
(該提問可能引用了上文內容)，
請將其改寫為一個獨立的、可被理解的問題。
不需要回答問題，只要重寫它，如果不需要重寫則保持原樣。
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

summary_system_prompt = """
你是一個對話整理助手。以下是一段與面試官的歷史對話，
請將它濃縮成一小段摘要，保留關於我（受面試者）的背景、經驗、專案與需求等重要資訊，
之後會用這個摘要幫助回答後續問題。請用自然的中文第一人稱敘述。
"""
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("human", "{history_text}"),
])

# 初始化向量資料庫與 chain（與原 app.py 相同邏輯）
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small"),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

history_aware_model = init_chat_model("gpt-4.1-mini", api_key=API_KEY)

# history_aware_retriever 負責處理 "追問" 的邏輯
history_aware_retriever = create_history_aware_retriever(
    history_aware_model, retriever, contextualize_q_prompt
)

qa_system_prompt = """
你是{MY_NAME}。請根據以下的上下文片段來回答面試官的問題。
如果你不知道答案，就說不知道，不要編造內容。
請保持專業且自信的語氣。

上下文: {context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"), # 這裡也會放入歷史，讓回答更有連續性
    ("human", "{input}"),
]).partial(MY_NAME=MY_NAME)

qa_assistant_llm = init_chat_model("gpt-4.1-mini", api_key=API_KEY)
question_answer_chain = create_stuff_documents_chain(qa_assistant_llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

FIRESTORE_CLIENT = firestore.Client(
    project=FIRESTORE_PROJECT_ID,
    database=FIRESTORE_DATABASE)

def get_session_history(session_id: str):
    if not session_id:
        raise ValueError("session_id 不可為空")
    logger.info("get_session_history start, session_id=%s", session_id)
    history = FirestoreChatMessageHistory(
        session_id=session_id,
        collection=FIRESTORE_COLLECTION,
        client=FIRESTORE_CLIENT,
    )
    logger.info("get_session_history done")
    return history

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding.encode(text))


def _estimate_history_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total += _count_tokens(content)
        else:
            # 非字串內容時，保守一點用 str() 估算
            total += _count_tokens(str(content))
    return total


def summarize_history_if_too_long(session_id: str) -> None:
    """
    若歷史對話的 token 數超過 MAX_HISTORY_TOKENS，
    則把較舊的訊息濃縮成一則 system summary，
    並保留最近 RECENT_MESSAGES_TO_KEEP 則原始訊息。
    """
    history = get_session_history(session_id)
    messages = history.messages

    total_tokens = _estimate_history_tokens(messages)
    logger.info(
        "Session %s history estimated tokens: %d (limit %d)",
        session_id,
        total_tokens,
        MAX_HISTORY_TOKENS,
    )

    if total_tokens <= MAX_HISTORY_TOKENS:
        return

    # 切出「要摘要的舊訊息」與「保留的最近幾則」
    if len(messages) <= RECENT_MESSAGES_TO_KEEP:
        old_messages = messages
        recent_messages = []
    else:
        old_messages = messages[:-RECENT_MESSAGES_TO_KEEP]
        recent_messages = messages[-RECENT_MESSAGES_TO_KEEP:]

    if not old_messages:
        return

    # 將舊訊息轉成文字餵給摘要模型
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

    logger.info(
        "Summarizing history for session %s: %d old messages, %d recent kept",
        session_id,
        len(old_messages),
        len(recent_messages),
    )

    # 使用較輕量的 qa_assistant_llm 進行摘要
    summary_chain = summary_prompt | qa_assistant_llm
    summary_msg = summary_chain.invoke({"history_text": history_text})
    summary_text = getattr(summary_msg, "content", "")

    summary_system_msg = SystemMessage(
        content=f"以下是更早期對話的摘要，供後續回答問題時參考：\n{summary_text}"
    )

    # 用「摘要 + 最近幾則原始訊息」覆寫歷史
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

@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="訊息不可為空白")

    logger.info("/chat start, session_id=%s, message=%s", req.session_id, req.message)

    try:
        result = conversational_rag_chain.invoke(
            {"input": req.message},
            config={"configurable": {"session_id": req.session_id}}
        )
        logger.info("conversational_rag_chain.invoke done, result_keys=%s", list(result.keys()))
    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e)) from e

    answer = result.get("answer", "抱歉，目前無法產生回應。")

     # 若歷史 token 過多，將較舊歷史濃縮成摘要
    try:
        summarize_history_if_too_long(req.session_id)
    except Exception:
        # 不讓摘要錯誤影響主要回應
        logger.exception("summarize_history_if_too_long failed")
    logger.info("/chat end, answer_len=%d", len(answer))
    return ChatResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
