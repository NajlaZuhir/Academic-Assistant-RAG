# 6) sends retrieved chunks + query to Mistral and returns an answer

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from dotenv import load_dotenv
from retrieval import load_embeddings, retrieve


embeddings_dir = "policy_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"   # free tier, fast
TOP_K = 3


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the prompt by injecting retrieved chunks as context."""
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i}: {chunk['policy_name']}]\n{chunk['text']}\n\n"

    return f"""You are a helpful university policy assistant for UDST (University of Doha for Science and Technology).
Answer the student's question using ONLY the policy excerpts provided below.
If the answer is not found in the excerpts, say "I couldn't find this in the available policies."

--- POLICY EXCERPTS ---
{context}
--- END OF EXCERPTS ---

Student question: {query}

Answer clearly and concisely, and mention which policy your answer is based on."""


def generate_answer(query: str, client: Mistral, chunks: list[dict]) -> str:
    """Send the prompt to Mistral and return the answer."""
    prompt = build_prompt(query, chunks)

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()
    
    # Load API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Set your MISTRAL_API_KEY environment variable first.")

    client = Mistral(api_key=api_key)

    print("Loading model and embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks = load_embeddings(embeddings_dir)
    print(f"Ready — {len(chunks)} chunks loaded\n")

    while True:
        query = input("Your question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break

        # Retrieve relevant chunks
        top_chunks = retrieve(query, model, embeddings, chunks, top_k=TOP_K)

        # Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(query, client, top_chunks)

        print("=" * 60)
        print(answer)
        print("=" * 60)

        # Show sources
        print("\nSources used:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"  [{i}] {chunk['policy_name']} (score: {chunk['score']})")
        print()