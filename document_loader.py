from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def load_uploaded_document(uploaded_file):
    """Read an uploaded PDF or TXT file and return its text."""

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        document_text = "\n\n".join(pages)

    elif extension == ".txt":
        document_text = uploaded_file.getvalue().decode(
            "utf-8-sig",
            errors="replace",
        )

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not document_text.strip():
        raise ValueError(
            f"No readable text found in {filename}. "
            "Scanned PDFs may require OCR."
        )

    return {
        "text": document_text,
        "source": filename,
    }