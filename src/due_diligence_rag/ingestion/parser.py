import json
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMode,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from due_diligence_rag.domain.document import (
    ParsedDocument,
    SourceDocument,
)


class DocumentParsingError(RuntimeError):
    """Raised when Docling cannot parse a document."""


class DoclingPDFParser:
    """Parse a PDF into Markdown and structured JSON using full-page OCR."""

    def __init__(
        self,
        output_directory: Path,
        converter: DocumentConverter | None = None,
    ) -> None:
        self.output_directory = output_directory.resolve()

        # Use an externally supplied converter when provided.
        # Otherwise, create the default full-page OCR converter.
        self.converter = converter or self._create_ocr_converter()

    @staticmethod
    def _create_ocr_converter() -> DocumentConverter:
        """Create a Docling converter configured for full-page EasyOCR."""

        ocr_options = EasyOcrOptions(
            mode=OcrMode.FULL_PAGE,
            lang=["en"],
        )

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def parse(self, source: SourceDocument) -> ParsedDocument:
        document_output_directory = self.output_directory / source.document_id
        document_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        markdown_path = document_output_directory / "document.md"
        json_path = document_output_directory / "document.json"
        metadata_path = document_output_directory / "metadata.json"

        try:
            conversion_result = self.converter.convert(source.file_path)
            document = conversion_result.document

            markdown = document.export_to_markdown()
            structured_data = self._export_structured_data(document)

            markdown_path.write_text(
                markdown,
                encoding="utf-8",
            )

            json_path.write_text(
                json.dumps(
                    structured_data,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            parsed_document = ParsedDocument(
                document_id=source.document_id,
                filename=source.filename,
                sha256=source.sha256,
                markdown_path=markdown_path,
                json_path=json_path,
                metadata_path=metadata_path,
            )

            metadata_path.write_text(
                parsed_document.model_dump_json(indent=2),
                encoding="utf-8",
            )

            return parsed_document

        except Exception as error:
            raise DocumentParsingError(
                f"Failed to parse PDF: {source.file_path}"
            ) from error

    @staticmethod
    def _export_structured_data(
        document: Any,
    ) -> dict[str, Any]:
        if hasattr(document, "export_to_dict"):
            result = document.export_to_dict()

            if isinstance(result, dict):
                return result

        if hasattr(document, "model_dump"):
            result = document.model_dump(mode="json")

            if isinstance(result, dict):
                return result

        raise DocumentParsingError(
            "The installed Docling version does not provide "
            "a supported JSON export method."
        )
