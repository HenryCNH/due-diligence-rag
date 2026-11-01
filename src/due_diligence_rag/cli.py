from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.panel import Panel

from due_diligence_rag.config import get_settings
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

            title = f"Chunk {index} | " f"tokens={token_count} | " f"pages={pages}"

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


if __name__ == "__main__":
    app()
