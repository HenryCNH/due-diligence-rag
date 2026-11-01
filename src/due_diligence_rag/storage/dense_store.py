import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from due_diligence_rag.domain.document import (
    DocumentChunk,
    RetrievalResult,
)


class DenseVectorStore:
    """Store and retrieve dense document-chunk embeddings."""

    VECTOR_NAME = "dense"

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name must not be empty.")
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be greater than zero.")

        self.client = client
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

    def ensure_collection(self) -> None:
        """Create the collection when it does not already exist."""
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.VECTOR_NAME: models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def upsert_chunks(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Insert or replace chunks and their dense vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must equal the number of embeddings."
            )

        for index, embedding in enumerate(embeddings):
            if len(embedding) != self.vector_dimension:
                raise ValueError(
                    "Embedding dimension mismatch at index "
                    f"{index}: expected {self.vector_dimension}, "
                    f"received {len(embedding)}."
                )

        points = [
            models.PointStruct(
                id=self._to_qdrant_id(chunk.chunk_id),
                vector={self.VECTOR_NAME: embedding},
                payload=self._chunk_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        *,
        query_embedding: list[float],
        limit: int,
    ) -> list[RetrievalResult]:
        """Return the most similar chunks."""
        if len(query_embedding) != self.vector_dimension:
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected {self.vector_dimension}, "
                f"received {len(query_embedding)}."
            )
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )

        results: list[RetrievalResult] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                RetrievalResult(
                    chunk_id=self._required_string(payload, "chunk_id"),
                    document_id=self._required_string(payload, "document_id"),
                    filename=self._required_string(payload, "filename"),
                    score=float(point.score),
                    text=self._required_string(payload, "text"),
                    contextualized_text=self._required_string(
                        payload,
                        "contextualized_text",
                    ),
                    headings=self._string_list(payload.get("headings")),
                    page_numbers=self._integer_list(payload.get("page_numbers")),
                    content_labels=self._string_list(payload.get("content_labels")),
                )
            )

        return results

    @staticmethod
    def _to_qdrant_id(chunk_id: str) -> str:
        """
        Convert a human-readable chunk ID into a deterministic UUID.

        Qdrant accepts unsigned integers or UUIDs as point IDs.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _chunk_payload(chunk: DocumentChunk) -> dict[str, object]:
        """Convert a document chunk into a Qdrant-compatible payload."""
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "filename": chunk.filename,
            "text": chunk.text,
            "contextualized_text": chunk.contextualized_text,
            "headings": chunk.headings,
            "page_numbers": chunk.page_numbers,
            "content_labels": chunk.content_labels,
            "source_references": chunk.source_references,
            "token_count": chunk.token_count,
        }

    @staticmethod
    def _required_string(payload: dict[str, object], key: str) -> str:
        """Read a required, non-null string-compatible payload value."""
        if key not in payload or payload[key] is None:
            raise ValueError(f"Qdrant payload is missing required field: {key!r}.")

        return str(payload[key])

    @staticmethod
    def _string_list(value: object) -> list[str]:
        """Convert a list-like payload value into a list of strings."""
        if not isinstance(value, list):
            return []

        return [str(item) for item in value]

    @staticmethod
    def _integer_list(value: object) -> list[int]:
        """Return integer items from a list-like payload value."""
        if not isinstance(value, list):
            return []

        return [
            item
            for item in value
            if isinstance(item, int) and not isinstance(item, bool)
        ]
