from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    raw_data_directory: Path = Path("data/raw")
    parsed_data_directory: Path = Path("data/parsed")

    maximum_pdf_size_mb: int = 100

    embedding_model_name: str = "BAAI/bge-m3"
    maximum_chunk_tokens: int = 512
    embedding_batch_size: int = 4
    embedding_device: Literal["auto", "cpu", "cuda"] = "cuda"

    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_local_path: Path = Path("data/qdrant")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "due_diligence_chunks"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached application settings object."""

    return Settings()
