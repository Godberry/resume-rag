"""FastAPI application entry point."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.graph.workflow import graph_app
from app.utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


# FastAPI application
app = FastAPI(
    title="Resume RAG API",
    description="基於履歷的 RAG 聊天後端 (LangGraph 版本)",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request payload."""

    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Chat response payload."""

    answer: str


# Endpoints
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Process chat message and return RAG response.

    Args:
        req: Chat request with message and session_id

    Returns:
        Chat response with generated answer

    Raises:
        HTTPException: If message is empty or processing fails
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="訊息不可為空白")

    logger.info("/chat start, session_id=%s, message=%s", req.session_id, req.message)

    try:
        # LangGraph invoke with session-based thread_id
        config = {"configurable": {"thread_id": req.session_id}}
        input_state = {"messages": [HumanMessage(content=req.message)]}

        # Execute graph
        result = graph_app.invoke(input_state, config=config)

        # Extract last message as response
        messages = result["messages"]
        answer = messages[-1].content

    except Exception as e:
        logger.exception("Error in /chat")
        raise HTTPException(status_code=500, detail=str(e)) from e

    logger.info("/chat end, answer_len=%d", len(answer))
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
