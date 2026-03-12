"""Application settings loaded from .env via pydantic BaseSettings.

Settings are lazily instantiated so that dotenv / environment variables
are already loaded by the time the object is first accessed.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve .env relative to this file, not the cwd
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    """Central configuration — all values come from environment / .env file."""

    # LLM (Google Gemini)
    GEMINI_API_KEY: str
    LLM_MODEL: str = "gemini-2.5-flash"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    COLLECTION_NAME: str = "hyde_rag_kb"

    # Embedding (Google)
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # Retrieval
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.6
    MAX_CONTEXT_TOKENS: int = 3000

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings()


class _SettingsProxy:
    """Lazy proxy — defers Settings() construction until first attribute access."""

    def __getattr__(self, name: str):
        return getattr(_get_settings(), name)


settings = _SettingsProxy()
