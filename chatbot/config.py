"""Project settings, loaded from the .env file at the project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchainvector")

# Must match the dimension of the Pinecone index (ada-002 -> 1536).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = 1536

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50


def require_keys() -> None:
    missing = [n for n, v in [("OPENAI_API_KEY", OPENAI_API_KEY), ("PINECONE_API_KEY", PINECONE_API_KEY)] if not v]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {PROJECT_ROOT / '.env'}")
