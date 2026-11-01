"""Dense embedding service backed by Sentence Transformers."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

Device = Literal["auto", "cpu", "cuda"]


class DenseEmbeddingService:
    """Create normalized dense embeddings with Sentence Transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        device: Device = "auto",
        batch_size: int = 4,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: Sentence Transformers model name or local model path.
            device: Device selection. ``auto`` uses CUDA when available.
            batch_size: Number of document chunks encoded per batch.

        Raises:
            ValueError: If ``model_name`` is empty or ``batch_size`` is invalid.
            RuntimeError: If CUDA is requested but unavailable, or if the model's
                embedding dimension cannot be determined.
        """
        cleaned_model_name = model_name.strip()
        if not cleaned_model_name:
            raise ValueError("The model name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = cleaned_model_name
        self.batch_size = batch_size
        self.device = self._resolve_device(device)

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name}."
            )

        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks in batches.

        Args:
            texts: Document chunks to embed.

        Returns:
            One normalized embedding vector for each input text.

        Raises:
            ValueError: If any document chunk is empty or contains only whitespace.
            RuntimeError: If the model returns an unexpected embedding shape.
        """
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts]
        empty_positions = [
            index for index, text in enumerate(cleaned_texts) if not text
        ]
        if empty_positions:
            raise ValueError(
                "Document chunks cannot be empty. "
                f"Empty chunk positions: {empty_positions}."
            )

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return self._to_lists(
            embeddings,
            expected_rows=len(cleaned_texts),
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query as a normalized vector.

        Args:
            query: Search query to embed.

        Returns:
            The normalized query embedding.

        Raises:
            ValueError: If the query is empty or contains only whitespace.
            RuntimeError: If the model returns an unexpected embedding shape.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("The search query cannot be empty.")

        embedding = np.asarray(
            self.model.encode(
                cleaned_query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        if embedding.ndim != 1 or embedding.shape[0] != self.dimension:
            raise RuntimeError(
                "The model returned an unexpected query embedding shape: "
                f"{embedding.shape}. Expected ({self.dimension},)."
            )

        return embedding.astype(float, copy=False).tolist()

    @staticmethod
    def _resolve_device(requested_device: Device) -> str:
        """Resolve an explicit or automatic PyTorch device selection."""
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: 'auto', 'cpu', or 'cuda'.")

        if requested_device == "cpu":
            return "cpu"

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but PyTorch cannot access a CUDA GPU."
                )
            return "cuda"

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _to_lists(
        self,
        embeddings: np.ndarray,
        *,
        expected_rows: int,
    ) -> list[list[float]]:
        """Validate a document embedding matrix and convert it to Python lists."""
        array = np.asarray(embeddings)
        expected_shape = (expected_rows, self.dimension)

        if array.ndim != 2 or array.shape != expected_shape:
            raise RuntimeError(
                "The model returned an unexpected document embedding shape: "
                f"{array.shape}. Expected {expected_shape}."
            )

        return array.astype(float, copy=False).tolist()
