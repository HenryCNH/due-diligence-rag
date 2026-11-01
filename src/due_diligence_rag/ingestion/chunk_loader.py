import json
from pathlib import Path

from due_diligence_rag.domain.document import DocumentChunk


class ChunkLoadingError(ValueError):
    """Raised when a chunk JSONL file is invalid."""


def load_chunks(chunks_path: Path) -> list[DocumentChunk]:
    """Load and validate DocumentChunk records from JSONL."""

    resolved_path = chunks_path.expanduser().resolve()

    if not resolved_path.exists():
        raise ChunkLoadingError(f"Chunks file does not exist: {resolved_path}")

    if not resolved_path.is_file():
        raise ChunkLoadingError(f"Chunks path is not a file: {resolved_path}")

    chunks: list[DocumentChunk] = []

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            try:
                record = json.loads(cleaned_line)
                chunks.append(DocumentChunk.model_validate(record))
            except (json.JSONDecodeError, ValueError) as error:
                raise ChunkLoadingError(
                    f"Invalid chunk record at line {line_number}: {resolved_path}"
                ) from error

    if not chunks:
        raise ChunkLoadingError(f"No chunks were found in: {resolved_path}")

    return chunks
