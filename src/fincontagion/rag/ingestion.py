
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_DIR = PROJECT_ROOT / "data" / "raw" / "financial_documents"


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract non-empty text from a PDF, retaining source metadata."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_absolute():
        pdf_path = DOCUMENT_DIR / pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


if __name__ == "__main__":
    pdf_name = "rbi_fsr_june_2025.pdf"
    extracted_pages = extract_pdf_pages(pdf_name)

    total_characters = sum(
        len(page["text"]) for page in extracted_pages
    )

    print(f"Source PDF: {pdf_name}")
    print(f"Pages containing text: {len(extracted_pages)}")
    print(f"Total extracted characters: {total_characters}")

    if extracted_pages:
        first_page = extracted_pages[0]
        print(f"\nFirst extracted page: {first_page['page']}")
        print("\nText preview:")
        print(first_page["text"][:1000])
    else:
        print("No extractable text found. OCR may be required.")
