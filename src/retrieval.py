# 5) finds the most relevant chunks for a user query

import json
import numpy as np
from sentence_transformers import SentenceTransformer

embeddings_dir = "policy_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5   # number of chunks to retrieve


def load_embeddings(embeddings_dir: str):
    """Load saved vectors and chunk metadata."""
    embeddings = np.load(f"{embeddings_dir}/embeddings.npy")
    with open(f"{embeddings_dir}/chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and all chunk vectors."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    corpus_norm = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    return corpus_norm @ query_norm


def retrieve(query: str, model: SentenceTransformer, embeddings: np.ndarray, chunks: list, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and return the top_k most similar chunks."""
    query_vec = model.encode(query)
    scores = cosine_similarity(query_vec, embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": round(float(scores[idx]), 4),
            "policy_name": chunks[idx]["policy_name"],
            "chunk_index": chunks[idx]["chunk_index"],
            "text": chunks[idx]["text"],
        })
    return results


if __name__ == "__main__":
    print("Loading model and embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks = load_embeddings(embeddings_dir)
    print(f"Ready — {len(chunks)} chunks loaded\n")

    while True:
        query = input("Enter your question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        results = retrieve(query, model, embeddings, chunks)

        print(f"\nTop {TOP_K} relevant chunks:\n" + "-" * 60)
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']}  |  Policy: {r['policy_name']}  |  Chunk: {r['chunk_index']}")
            print(f"    {r['text'][:300]}...")
            print() 