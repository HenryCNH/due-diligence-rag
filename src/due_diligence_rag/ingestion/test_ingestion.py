from pathlib import Path

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

pdf_path = r"C:\Users\ngaih\OneDrive\Desktop\CUHK_Cloud\Self_Study\due-diligence-rag\data\raw\261012_CBI_Employment_Screening_EN.pdf"


pdf_path = Path(pdf_path)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True

pipeline_options.ocr_options = EasyOcrOptions(
    mode=OcrMode.FULL_PAGE,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

result = converter.convert(pdf_path)
markdown = result.document.export_to_markdown()

print(markdown)
