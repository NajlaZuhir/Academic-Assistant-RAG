# 4) turns chunks into vectors and stores them

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

chunks_dir = "policy_chunks"
embeddings_dir = "policy_embeddings"
os.makedirs(embeddings_dir, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight, good for semantic search


def load_all_chunks(chunks_dir: str) -> list[dict]:
    """Load the combined chunks file."""
    combined_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(combined_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed all chunk texts into vectors."""
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


if __name__ == "__main__":
    # Load chunks
    chunks = load_all_chunks(chunks_dir)
    print(f"Loaded {len(chunks)} chunks\n")

    # Load model and embed
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)

    # Save embeddings as .npy
    embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Saved embeddings → {embeddings_path}")

    # Save chunk metadata alongside (so retrieval can match vector → chunk)
    metadata_path = os.path.join(embeddings_dir, "chunks_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata   → {metadata_path}")