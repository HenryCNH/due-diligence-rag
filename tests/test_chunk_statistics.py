from due_diligence_rag.domain.document import DocumentChunk
from due_diligence_rag.ingestion.chunk_statistics import (
    calculate_chunk_statistics,
)


def test_calculate_chunk_statistics() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="doc-1:chunk:00000",
            document_id="doc-1",
            chunk_index=0,
            filename="guideline.pdf",
            text="Required document one.",
            contextualized_text=("Required Documents\nRequired document one."),
            headings=["Required Documents"],
            page_numbers=[1],
            content_labels=["text"],
            source_references=["#/texts/1"],
            token_count=10,
        ),
        DocumentChunk(
            chunk_id="doc-1:chunk:00001",
            document_id="doc-1",
            chunk_index=1,
            filename="guideline.pdf",
            text="Required document two.",
            contextualized_text=("Required Documents\nRequired document two."),
            headings=["Required Documents"],
            page_numbers=[1, 2],
            content_labels=["text", "list_item"],
            source_references=["#/texts/2"],
            token_count=20,
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics["number_of_chunks"] == 2
    assert statistics["minimum_tokens"] == 10
    assert statistics["maximum_tokens"] == 20
    assert statistics["average_tokens"] == 15
    assert statistics["chunks_without_page_numbers"] == 0
    assert statistics["chunks_with_multiple_pages"] == 1
    assert statistics["content_labels"]["text"] == 2


def test_empty_chunk_statistics() -> None:
    statistics = calculate_chunk_statistics([])

    assert statistics["number_of_chunks"] == 0
    assert statistics["average_tokens"] == 0.0
    assert statistics["chunks_without_page_numbers"] == 0
