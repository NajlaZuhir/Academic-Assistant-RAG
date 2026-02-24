# chunking.py — chunks academic catalog pages into overlapping sections

import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

catalog_dir = "catalog_data"
chunks_dir = "catalog_chunks"
os.makedirs(chunks_dir, exist_ok=True)

CHUNK_SIZE = 500 # max characters per chunk
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SIMILARITY_THRESHOLD = 0.85


def load_pages(catalog_dir: str) -> list[dict]:
    """Load extracted catalog pages from JSON."""
    path = os.path.join(catalog_dir, "catalog_pages.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noise."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', ' ', text)       # remove "......" leaders (common in PDF tables of contents)
    text = re.sub(r'\b\d{1,3}\b(?=\s*$)', '', text)  # remove lone page numbers at end of line
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    """Split text using LangChain RecursiveCharacterTextSplitter.
    "Recursive" tries each separator in order"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def merge_semantic_chunks(chunks: list[str], model: SentenceTransformer) -> list[str]:
    """Merge consecutive chunks that are semantically similar."""
    if len(chunks) <= 1:
        return chunks

    embeddings = model.encode(chunks, show_progress_bar=False) # # turn each chunk into a vector
    merged = [chunks[0]]

    for i in range(1, len(chunks)):
        sim = np.dot(embeddings[i], embeddings[len(merged) - 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[len(merged) - 1]) + 1e-10
        )
        if sim > SIMILARITY_THRESHOLD:
            merged[-1] += " " + chunks[i]
        else:
            merged.append(chunks[i])

    return merged


def chunk_catalog(pages: list[dict], model: SentenceTransformer) -> list[dict]:
    """Process all pages into chunks with page metadata."""
    all_chunks = []
    chunk_id = 0

    for page in pages: # orchestrator — loops every page through the 3 steps
        text = clean_text(page["text"])
        if not text or len(text) < 50:      # skip near-empty pages
            continue

        initial_chunks = split_into_chunks(text)
        final_chunks = merge_semantic_chunks(initial_chunks, model)

        for chunk in final_chunks: # metadata
            all_chunks.append({
                "chunk_id": f"catalog__chunk_{chunk_id}",
                "page_number": page["page_number"],
                "chunk_index": chunk_id,
                "text": chunk,
            })
            chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading catalog pages...")
    pages = load_pages(catalog_dir)
    print(f"Loaded {len(pages)} pages\n")

    print("Chunking...")
    all_chunks = chunk_catalog(pages, model)

    # Save combined chunks
    output_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved → {output_path}")
    print(f"\nSample chunk:\n{all_chunks[0]['text'][:300]}")