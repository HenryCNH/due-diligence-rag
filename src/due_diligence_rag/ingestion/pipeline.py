import json
from pathlib import Path

import structlog

from due_diligence_rag.domain.document import ParsedDocument
from due_diligence_rag.ingestion.chunk_statistics import (
    calculate_chunk_statistics,
)
from due_diligence_rag.ingestion.chunker import (
    DoclingDocumentChunker,
)
from due_diligence_rag.ingestion.file_validator import (
    PDFFileValidator,
)
from due_diligence_rag.ingestion.parser import DoclingPDFParser

logger = structlog.get_logger(__name__)


# class PDFIngestionPipeline:
#     """Coordinate PDF validation and parsing."""

#     def __init__(
#         self,
#         validator: PDFFileValidator,
#         parser: DoclingPDFParser,
#     ) -> None:
#         self.validator = validator
#         self.parser = parser

#     def ingest(self, file_path: Path) -> ParsedDocument:
#         logger.info(
#             "document_ingestion_started",
#             file_path=str(file_path),
#         )

#         source_document = self.validator.validate(file_path)

#         logger.info(
#             "document_validation_completed",
#             document_id=source_document.document_id,
#             filename=source_document.filename,
#             file_size_bytes=source_document.file_size_bytes,
#         )

#         parsed_document = self.parser.parse(source_document)

#         logger.info(
#             "document_parsing_completed",
#             document_id=parsed_document.document_id,
#             markdown_path=str(parsed_document.markdown_path),
#             json_path=str(parsed_document.json_path),
#         )

#         return parsed_document


# ===============================v2==========================


class PDFIngestionPipeline:
    """Validate, parse and chunk one PDF document."""

    def __init__(
        self,
        validator: PDFFileValidator,
        parser: DoclingPDFParser,
        chunker: DoclingDocumentChunker,
    ) -> None:
        self.validator = validator
        self.parser = parser
        self.chunker = chunker

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

        parsing_result = self.parser.parse(source_document)
        parsed_document = parsing_result.parsed_document

        chunks = self.chunker.chunk_document(
            document=parsing_result.docling_document,
            document_id=source_document.document_id,
            filename=source_document.filename,
        )

        document_output_directory = parsed_document.markdown_path.parent
        chunks_path = document_output_directory / "chunks.jsonl"
        statistics_path = document_output_directory / "chunk_statistics.json"

        self._write_chunks(
            chunks_path=chunks_path,
            chunks=chunks,
        )

        statistics = calculate_chunk_statistics(chunks)

        statistics_path.write_text(
            json.dumps(
                statistics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(
            "document_chunking_completed",
            document_id=source_document.document_id,
            number_of_chunks=len(chunks),
            chunks_path=str(chunks_path),
            statistics_path=str(statistics_path),
        )

        return parsed_document

    @staticmethod
    def _write_chunks(
        *,
        chunks_path: Path,
        chunks: list,
    ) -> None:
        with chunks_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for chunk in chunks:
                output_file.write(chunk.model_dump_json())
                output_file.write("\n")
