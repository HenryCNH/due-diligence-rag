from due_diligence_rag.config import Settings
from due_diligence_rag.embeddings.dense import DenseEmbeddingService
from due_diligence_rag.storage.dense_store import DenseVectorStore
from due_diligence_rag.storage.qdrant_client import (
    create_qdrant_client,
)


def create_dense_components(
    settings: Settings,
) -> tuple[DenseEmbeddingService, DenseVectorStore]:
    """Create the dense embedding and storage components."""

    embedding_service = DenseEmbeddingService(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )

    client = create_qdrant_client(
        mode=settings.qdrant_mode,
        local_path=settings.qdrant_local_path,
        url=settings.qdrant_url,
    )

    vector_store = DenseVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        vector_dimension=embedding_service.dimension,
    )
    vector_store.ensure_collection()

    return embedding_service, vector_store
