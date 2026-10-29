from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Information about a source PDF before parsing."""

    document_id: str
    filename: str
    file_path: Path
    sha256: str
    file_size_bytes: int
    media_type: str = "application/pdf"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ParsedDocument(BaseModel):
    """Information about the files produced by the parser."""

    document_id: str
    filename: str
    sha256: str
    markdown_path: Path
    json_path: Path
    metadata_path: Path
    parser_name: str = "docling"
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
