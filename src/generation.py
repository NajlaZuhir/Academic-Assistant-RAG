# 6) sends retrieved chunks + query to LLM and returns an answer

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from .retrieval import load_embeddings, retrieve

embeddings_dir = "catalog_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TOP_K = 6


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i} — Page {chunk['page_number']}]\n{chunk['text']}\n\n"

    return f"""You are a helpful academic assistant for UDST (University of Doha for Science and Technology).
Answer the student's question using ONLY the catalog excerpts provided below.
If the answer is not found in the excerpts, say "I couldn't find this in the academic catalog."
Always mention the page number your answer is based on.

--- CATALOG EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Student question: {query}

Answer clearly and concisely."""


def generate_answer(query: str, client: InferenceClient, chunks: list[dict]) -> str:
    prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set your HF_TOKEN in the .env file.")

    client = InferenceClient(token=hf_token)

    print("Loading embedding model and embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks = load_embeddings(embeddings_dir)
    print(f"Ready — {len(chunks)} chunks loaded\n")

    while True:
        query = input("Your question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        top_chunks = retrieve(query, model, embeddings, chunks, top_k=TOP_K)

        print("\nGenerating answer...\n")
        answer = generate_answer(query, client, top_chunks)

        print("=" * 60)
        print(answer)
        print("=" * 60)

        print("\nSources used:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"  [{i}] Page {chunk['page_number']} (score: {chunk['score']})")
        print()