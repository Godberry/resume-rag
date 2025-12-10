# backend/config.py
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resume-rag")

FIRESTORE_PROJECT_ID = os.getenv("FIRESTORE_PROJECT_ID")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "(default)")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "chat_sessions")

MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))
RECENT_MESSAGES_TO_KEEP = int(os.getenv("RECENT_MESSAGES_TO_KEEP", "6"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOKEN_ENCODING_MODEL = os.getenv("TOKEN_ENCODING_MODEL", "gpt-4o-mini")
QA_CHAT_MODEL = os.getenv("QA_CHAT_MODEL", "gpt-4.1-mini")


MY_NAME = "許皓翔"


def validate_settings() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 環境變數未設定。")
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY 環境變數未設定。")
    if not FIRESTORE_PROJECT_ID:
        raise RuntimeError("FIRESTORE_PROJECT_ID 環境變數未設定。")
    logger.info("Config validated successfully")