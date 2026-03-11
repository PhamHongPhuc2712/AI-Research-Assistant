"""
config.py — Application configuration via environment variables.

Loads all settings from a .env file using pydantic-settings BaseSettings.
Exports a module-level `settings` singleton for use throughout the backend.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the project root as the parent of the backend/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    All fields have sensible defaults except OPENAI_API_KEY, which must be
    provided via the .env file or the environment.

    Attributes:
        OPENAI_API_KEY: OpenAI API key (required).
        LLM_MODEL: Model name for chat completions.
        EMBEDDING_MODEL: Model name for text embeddings.
        CHROMA_PATH: Filesystem path for ChromaDB persistent storage.
        CHUNK_SIZE: Number of characters per text chunk during ingestion.
        CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks.
        TOP_K_RETRIEVAL: Number of top results to return from retrieval.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHROMA_PATH: str = "./data/vectorstore"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5


# Module-level singleton — import this wherever settings are needed:
#   from backend.config import settings
settings = Settings()
