import hashlib
from pathlib import Path

from due_diligence_rag.domain.document import SourceDocument


class DocumentValidationError(ValueError):
    """Raised when a source document is invalid."""


class PDFFileValidator:
    """Validate a PDF and calculate its SHA-256 checksum."""

    PDF_SIGNATURE = b"%PDF-"

    def __init__(self, maximum_size_mb: int = 100) -> None:
        self.maximum_size_bytes = maximum_size_mb * 1024 * 1024

    def validate(self, file_path: Path) -> SourceDocument:
        resolved_path = file_path.expanduser().resolve()

        if not resolved_path.exists():
            raise DocumentValidationError(f"File does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise DocumentValidationError(f"Path is not a file: {resolved_path}")

        if resolved_path.suffix.lower() != ".pdf":
            raise DocumentValidationError(
                f"Only PDF files are supported: {resolved_path.name}"
            )

        file_size = resolved_path.stat().st_size

        if file_size == 0:
            raise DocumentValidationError("The PDF is empty.")

        if file_size > self.maximum_size_bytes:
            raise DocumentValidationError(
                f"The PDF exceeds the {self.maximum_size_bytes} byte limit."
            )

        with resolved_path.open("rb") as file:
            signature = file.read(len(self.PDF_SIGNATURE))

        if signature != self.PDF_SIGNATURE:
            raise DocumentValidationError(
                "The file has a .pdf extension but does not have a valid PDF signature."
            )

        sha256 = self._calculate_sha256(resolved_path)

        return SourceDocument(
            document_id=sha256,
            filename=resolved_path.name,
            file_path=resolved_path,
            sha256=sha256,
            file_size_bytes=file_size,
        )

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        digest = hashlib.sha256()

        with file_path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)

        return digest.hexdigest()
