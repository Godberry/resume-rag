"""FastAPI backend for resume RAG chatbot."""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config
from rag import conversational_rag_chain
from history import summarize_history_if_too_long

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config.validate_settings()

app = FastAPI(
    title="Resume RAG API",
    description="基於履歷的 RAG 聊天後端",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str


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
            config={"configurable": {"session_id": req.session_id}},
        )
        logger.info(
            "conversational_rag_chain.invoke done, result_keys=%s", list(result.keys())
        )
    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e)) from e

    answer = result.get("answer", "抱歉，目前無法產生回應。")

    try:
        summarize_history_if_too_long(req.session_id)
    except Exception:
        logger.exception("summarize_history_if_too_long failed")

    logger.info("/chat end, answer_len=%d", len(answer))
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
