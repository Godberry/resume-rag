"""FastAPI backend for resume RAG chatbot."""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-rag")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 環境變數未設定。")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY 環境變數未設定。")

MY_NAME = "許皓翔"

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


class ChatResponse(BaseModel):
    answer: str
    # 未來如果要回傳來源片段，可在這裡擴充
    # sources: List[str] = []


# 初始化向量資料庫與 chain（與原 app.py 相同邏輯）
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=OpenAIEmbeddings(api_key=API_KEY),
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(api_key=API_KEY, model_name="gpt-4.1-mini")

template = """
你是 {MY_NAME}。請根據底下的資訊回答面試官的問題。
如果資訊中沒有答案，請誠實回答「這在履歷中沒有提到，但我可以補充...」
請保持專業、自信且友善的語氣。

相關履歷資訊：
{context}

面試官問題：
{input}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template).partial(MY_NAME=MY_NAME)

combine_docs_chain = create_stuff_documents_chain(
    llm, QA_CHAIN_PROMPT
)

qa_chain = create_retrieval_chain(retriever, combine_docs_chain)


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="訊息不可為空白")

    try:
        result = qa_chain.invoke({"input": req.message})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    answer = result.get("answer", "抱歉，目前無法產生回應。")
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
