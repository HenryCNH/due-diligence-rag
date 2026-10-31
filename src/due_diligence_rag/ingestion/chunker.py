from collections.abc import Iterable
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer

from due_diligence_rag.domain.document import DocumentChunk


class DoclingDocumentChunker:
    """Create retrieval-ready chunks from a Docling document."""

    def __init__(
        self,
        embedding_model_name: str,
        maximum_chunk_tokens: int,
    ) -> None:
        hugging_face_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=hugging_face_tokenizer,
            max_tokens=maximum_chunk_tokens,
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
            repeat_table_header=True,
        )

    def chunk_document(
        self,
        *,
        document: Any,
        document_id: str,
        filename: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for index, chunk in enumerate(self.chunker.chunk(dl_doc=document)):
            text = chunk.text.strip()
            contextualized_text = self.chunker.contextualize(chunk=chunk).strip()

            if not text:
                continue

            page_numbers = self._extract_page_numbers(chunk)
            headings = self._string_list(getattr(chunk.meta, "headings", None))
            captions = self._string_list(getattr(chunk.meta, "captions", None))
            content_labels = self._extract_content_labels(chunk)
            source_references = self._extract_source_references(chunk)
            chunk_id = f"{document_id}:chunk:{index:05d}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=index,
                    filename=filename,
                    text=text,
                    contextualized_text=contextualized_text,
                    headings=headings,
                    captions=captions,
                    page_numbers=page_numbers,
                    content_labels=content_labels,
                    source_references=source_references,
                    token_count=self.tokenizer.count_tokens(contextualized_text),
                )
            )

        return chunks

    @staticmethod
    def _extract_page_numbers(chunk: Any) -> list[int]:
        page_numbers: set[int] = set()
        document_items = getattr(chunk.meta, "doc_items", [])

        for item in document_items:
            provenance_entries = getattr(item, "prov", [])
            for provenance in provenance_entries:
                page_number = getattr(provenance, "page_no", None)
                if isinstance(page_number, int):
                    page_numbers.add(page_number)

        return sorted(page_numbers)

    @staticmethod
    def _extract_content_labels(chunk: Any) -> list[str]:
        labels: set[str] = set()
        document_items = getattr(chunk.meta, "doc_items", [])

        for item in document_items:
            label = getattr(item, "label", None)
            if label is None:
                continue

            label_value = getattr(label, "value", str(label))
            labels.add(str(label_value))

        return sorted(labels)

    @staticmethod
    def _extract_source_references(chunk: Any) -> list[str]:
        references: set[str] = set()
        document_items = getattr(chunk.meta, "doc_items", [])

        for item in document_items:
            self_reference = getattr(item, "self_ref", None)
            if self_reference:
                references.add(str(self_reference))

        return sorted(references)

    @staticmethod
    def _string_list(values: Iterable[Any] | None) -> list[str]:
        if values is None:
            return []

        return [str(value).strip() for value in values if str(value).strip()]
