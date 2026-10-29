from pathlib import Path

import structlog

from due_diligence_rag.domain.document import ParsedDocument
from due_diligence_rag.ingestion.file_validator import PDFFileValidator
from due_diligence_rag.ingestion.parser import DoclingPDFParser

logger = structlog.get_logger(__name__)


class PDFIngestionPipeline:
    """Coordinate PDF validation and parsing."""

    def __init__(
        self,
        validator: PDFFileValidator,
        parser: DoclingPDFParser,
    ) -> None:
        self.validator = validator
        self.parser = parser

    def ingest(self, file_path: Path) -> ParsedDocument:
        logger.info(
            "document_ingestion_started",
            file_path=str(file_path),
        )

        source_document = self.validator.validate(file_path)

        logger.info(
            "document_validation_completed",
            document_id=source_document.document_id,
            filename=source_document.filename,
            file_size_bytes=source_document.file_size_bytes,
        )

        parsed_document = self.parser.parse(source_document)

        logger.info(
            "document_parsing_completed",
            document_id=parsed_document.document_id,
            markdown_path=str(parsed_document.markdown_path),
            json_path=str(parsed_document.json_path),
        )

        return parsed_document
