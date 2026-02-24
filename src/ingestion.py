# ingestion.py — download PDF and extract text by section

import requests
import json
import os
from pypdf import PdfReader
from io import BytesIO

PDF_URL = "https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf"
output_dir = "catalog_data"
os.makedirs(output_dir, exist_ok=True)

# Ingestion
def download_pdf(url: str) -> BytesIO:
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30, stream = True)
    response.raise_for_status()
    print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")
    return BytesIO(response.content)


def extract_pages(pdf_bytes: BytesIO) -> list[dict]:
    """Extract text page by page, keeping page number as metadata."""
    reader = PdfReader(pdf_bytes)
    pages = []
    print(f"Total pages: {len(reader.pages)}")

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text.strip()
            })

    return pages


def save_pages(pages: list[dict]):
    output_path = os.path.join(output_dir, "catalog_pages.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pages)} pages → {output_path}")


if __name__ == "__main__":
    pdf_bytes = download_pdf(PDF_URL)
    pages = extract_pages(pdf_bytes)
    save_pages(pages)
    print(f"\nDone — {len(pages)} pages extracted")
    print(f"Preview of page 1:\n{pages[8]['text'][:500]}")