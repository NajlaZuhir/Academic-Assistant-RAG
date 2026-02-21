# 4) Embeds chunked documents 

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from chunking import chunk_documents

MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return np.array(embeddings), chunks


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    print("Generating chunks...")
    chunks = chunk_documents()

    print("Generating embeddings...")
    embeddings, chunk_metadata = generate_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\nIndex built successfully!")
    print(f"Total chunks indexed: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")