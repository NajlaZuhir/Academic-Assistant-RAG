# 3) splits PDF text into overlapping chunks

import os
import re
import json
from pypdf import PdfReader

pdf_dir = "policy_pdfs"
chunks_dir = "policy_chunks"
os.makedirs(chunks_dir, exist_ok=True)

CHUNK_SIZE = 500        # number of words per chunk
CHUNK_OVERLAP = 50      # words shared between consecutive chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file """
    reader = PdfReader(pdf_path) # Opens the PDF
    full_text = ""
    for page in reader.pages:  #loops through every page of a PDF
        full_text += page.extract_text() + "\n"  # concatenates all the text into one big string
    return full_text.strip()


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # slide forward, keeping overlap

    return chunks


def chunk_policy(pdf_path: str, policy_name: str) -> list[dict]:
    """Extract text from a PDF and return a list of chunk dicts."""

    print(f"  Chunking: {policy_name}")

    raw_text = extract_text_from_pdf(pdf_path)
    text = clean_text(raw_text)
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Attach metadata to each chunk — important for retrieval later
    return [
        {
            "chunk_id": f"{policy_name}__chunk_{i}",
            "policy_name": policy_name,
            "source": pdf_path,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    all_chunks = []

    pdf_files = [f for f in os.listdir(pdf_dir)]
    print(f"Found {len(pdf_files)} PDFs in '{pdf_dir}'\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        policy_name = pdf_file.replace(".pdf", "").replace("_", " ")

        chunks = chunk_policy(pdf_path, policy_name) # chunk one policy
        all_chunks.extend(chunks) # add to master list

        # Save per-policy chunks as JSON
        chunk_path = os.path.join(chunks_dir, pdf_file.replace(".pdf", ".json"))
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"  → {len(chunks)} chunks saved to {chunk_path}\n")

    # Also save one combined file for easy loading later
    combined_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total Policies: {len(pdf_files)}")
    print(f"Total chunks across all policies: {len(all_chunks)}")
    print(f"Combined file saved to: {combined_path}")