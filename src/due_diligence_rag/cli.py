from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.panel import Panel

from due_diligence_rag.config import get_settings
from due_diligence_rag.ingestion.chunk_loader import load_chunks
from due_diligence_rag.ingestion.chunker import (
    DoclingDocumentChunker,
)
from due_diligence_rag.ingestion.file_validator import (
    PDFFileValidator,
)
from due_diligence_rag.ingestion.parser import DoclingPDFParser
from due_diligence_rag.ingestion.pipeline import (
    PDFIngestionPipeline,
)
from due_diligence_rag.retrieval.dense_retriever import DenseRetriever
from due_diligence_rag.retrieval.factory import create_dense_components

app = typer.Typer(
    name="diligence-rag",
    help="Due-diligence RAG development command-line interface.",
    no_args_is_help=True,
)

console = Console()


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


@app.command()
def ingest(
    pdf_path: Annotated[
        Path,
        typer.Argument(help="Path to the PDF that should be ingested."),
    ],
) -> None:
    """Validate, parse and chunk one PDF document."""

    configure_logging()
    settings = get_settings()

    validator = PDFFileValidator(maximum_size_mb=settings.maximum_pdf_size_mb)

    parser = DoclingPDFParser(output_directory=settings.parsed_data_directory)

    chunker = DoclingDocumentChunker(
        embedding_model_name=(settings.embedding_model_name),
        maximum_chunk_tokens=(settings.maximum_chunk_tokens),
    )

    pipeline = PDFIngestionPipeline(
        validator=validator,
        parser=parser,
        chunker=chunker,
    )

    result = pipeline.ingest(pdf_path)

    output_directory = result.markdown_path.parent

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]PDF ingested successfully[/bold green]",
                    f"Document ID: {result.document_id}",
                    f"Output: {output_directory}",
                    f"Markdown: {result.markdown_path.name}",
                    f"JSON: {result.json_path.name}",
                    "Chunks: chunks.jsonl",
                    "Statistics: chunk_statistics.json",
                ]
            ),
            title="Ingestion result",
        )
    )


@app.command("inspect-chunks")
def inspect_chunks(
    chunks_path: Annotated[
        Path,
        typer.Argument(help="Path to a chunks.jsonl file."),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of chunks to display.",
        ),
    ] = 5,
) -> None:
    """Display several generated chunks."""

    import json

    if not chunks_path.exists():
        raise typer.BadParameter(f"File does not exist: {chunks_path}")

    if limit < 1:
        raise typer.BadParameter("Limit must be at least 1.")

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for index, line in enumerate(input_file):
            if index >= limit:
                break

            record = json.loads(line)

            headings = record.get("headings", [])
            pages = record.get("page_numbers", [])
            token_count = record.get("token_count", 0)
            contextualized_text = record.get(
                "contextualized_text",
                "",
            )

            title = f"Chunk {index} | tokens={token_count} | pages={pages}"

            console.print(
                Panel(
                    "\n".join(
                        [
                            f"Headings: {headings}",
                            "",
                            contextualized_text,
                        ]
                    ),
                    title=title,
                )
            )


@app.command()
def index(
    chunks_path: Annotated[
        Path,
        typer.Argument(help="Path to the chunks.jsonl file to index."),
    ],
) -> None:
    """Create dense embeddings and store them in Qdrant."""

    settings = get_settings()
    chunks = load_chunks(chunks_path)

    embedding_service, vector_store = create_dense_components(settings)

    console.print(
        f"Embedding {len(chunks)} chunks on [bold]{embedding_service.device}[/bold]..."
    )

    embeddings = embedding_service.embed_documents(
        [chunk.contextualized_text for chunk in chunks]
    )

    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    vector_store.client.close()

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]Dense indexing completed[/bold green]",
                    f"Chunks indexed: {len(chunks)}",
                    f"Model: {embedding_service.model_name}",
                    f"Device: {embedding_service.device}",
                    f"Dimension: {embedding_service.dimension}",
                    f"Collection: {settings.qdrant_collection_name}",
                ]
            ),
            title="Index result",
        )
    )


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(help="Question used for semantic retrieval."),
    ],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            max=50,
            help="Maximum number of chunks to return.",
        ),
    ] = 5,
) -> None:
    """Search indexed chunks using dense retrieval."""

    settings = get_settings()

    embedding_service, vector_store = create_dense_components(settings)

    retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        vector_store.client.close()
        return

    for rank, result in enumerate(results, start=1):
        title = f"Rank {rank} | score={result.score:.4f} | pages={result.page_numbers}"

        body = "\n".join(
            [
                f"File: {result.filename}",
                f"Headings: {' > '.join(result.headings)}",
                f"Chunk ID: {result.chunk_id}",
                "",
                result.text,
            ]
        )

        console.print(
            Panel(
                body,
                title=title,
                border_style="cyan",
            )
        )

    vector_store.client.close()


if __name__ == "__main__":
    app()
