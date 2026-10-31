from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    raw_data_directory: Path = Path("data/raw")
    parsed_data_directory: Path = Path("data/parsed")

    maximum_pdf_size_mb: int = 100

    embedding_model_name: str = "BAAI/bge-m3"
    maximum_chunk_tokens: int = 512

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
