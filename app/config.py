"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str

    # Pinecone
    PINECONE_INDEX_NAME: str = "resume-rag"

    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    QA_CHAT_MODEL: str = "openai:gpt-4o-mini"

    # Memory Management
    RECENT_MESSAGES_TO_KEEP: int = 6

    # Personal Info
    MY_NAME: str = "許皓翔"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
