"""Vector store and retriever service."""

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Get the Pinecone vector store retriever.

    Uses LRU cache to ensure singleton pattern.
    """
    vector_store = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        pinecone_api_key=settings.PINECONE_API_KEY,
        embedding=OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL,
        ),
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})
