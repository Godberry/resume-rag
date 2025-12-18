"""LLM factory for creating chat model instances."""

from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Get the main LLM instance for QA and generation.

    Uses LRU cache to ensure singleton pattern.
    """
    return init_chat_model(settings.QA_CHAT_MODEL, api_key=settings.OPENAI_API_KEY)
