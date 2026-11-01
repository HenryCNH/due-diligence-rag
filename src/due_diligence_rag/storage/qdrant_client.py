from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient


def create_qdrant_client(
    *,
    mode: Literal["local", "server"],
    local_path: Path,
    url: str,
) -> QdrantClient:
    """Create either a local or server-backed Qdrant client."""

    if mode == "local":
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return QdrantClient(
            path=str(local_path),
        )

    if mode == "server":
        return QdrantClient(
            url=url,
        )

    raise ValueError(f"Unsupported Qdrant mode: {mode}")
