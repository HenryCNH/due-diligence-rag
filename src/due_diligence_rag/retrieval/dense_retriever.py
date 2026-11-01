from due_diligence_rag.domain.document import RetrievalResult
from due_diligence_rag.embeddings.dense import DenseEmbeddingService
from due_diligence_rag.storage.dense_store import DenseVectorStore


class DenseRetriever:
    """Retrieve document chunks using dense semantic similarity."""

    def __init__(
        self,
        *,
        embedding_service: DenseEmbeddingService,
        vector_store: DenseVectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")

        query_embedding = self.embedding_service.embed_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
        )
