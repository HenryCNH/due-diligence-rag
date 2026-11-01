from collections import Counter
from typing import Any

from due_diligence_rag.domain.document import DocumentChunk


def calculate_chunk_statistics(
    chunks: list[DocumentChunk],
) -> dict[str, Any]:
    """Calculate summary statistics for manual chunk inspection."""

    if not chunks:
        return {
            "number_of_chunks": 0,
            "minimum_tokens": 0,
            "maximum_tokens": 0,
            "average_tokens": 0.0,
            "chunks_without_page_numbers": 0,
            "content_labels": {},
        }

    token_counts = [chunk.token_count for chunk in chunks]

    label_counter: Counter[str] = Counter()

    for chunk in chunks:
        label_counter.update(chunk.content_labels)

    return {
        "number_of_chunks": len(chunks),
        "minimum_tokens": min(token_counts),
        "maximum_tokens": max(token_counts),
        "average_tokens": round(
            sum(token_counts) / len(token_counts),
            2,
        ),
        "chunks_without_page_numbers": sum(not chunk.page_numbers for chunk in chunks),
        "chunks_with_multiple_pages": sum(
            len(chunk.page_numbers) > 1 for chunk in chunks
        ),
        "content_labels": dict(sorted(label_counter.items())),
    }
