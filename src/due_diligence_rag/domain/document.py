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


class DocumentChunk(BaseModel):
    """One retrieval-ready section of a parsed document."""

    chunk_id: str
    document_id: str
    chunk_index: int

    filename: str

    text: str
    contextualized_text: str

    headings: list[str] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)

    content_labels: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)

    token_count: int

    parser_name: str = "docling"
    chunker_name: str = "docling_hybrid"
