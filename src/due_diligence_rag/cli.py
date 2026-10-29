from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.panel import Panel

from due_diligence_rag.ingestion.file_validator import PDFFileValidator
from due_diligence_rag.ingestion.parser import DoclingPDFParser
from due_diligence_rag.ingestion.pipeline import PDFIngestionPipeline

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
def parse(
    pdf_path: Annotated[
        Path,
        typer.Argument(help="Path to the PDF that should be parsed."),
    ],
) -> None:
    """Validate and parse one PDF document."""

    configure_logging()

    validator = PDFFileValidator(maximum_size_mb=100)

    parser = DoclingPDFParser(output_directory=Path("data/parsed"))

    pipeline = PDFIngestionPipeline(
        validator=validator,
        parser=parser,
    )

    result = pipeline.ingest(pdf_path)

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold green]PDF parsed successfully[/bold green]",
                    f"Document ID: {result.document_id}",
                    f"Markdown: {result.markdown_path}",
                    f"JSON: {result.json_path}",
                    f"Metadata: {result.metadata_path}",
                ]
            ),
            title="Ingestion result",
        )
    )


if __name__ == "__main__":
    app()
