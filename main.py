import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from src.ingestion import download_pdf, extract_pages, save_pages
from src.chunking import load_pages, chunk_catalog
from src.embedding import load_all_chunks, embed_chunks
from src.retrieval import load_embeddings, retrieve
from src.generation import build_prompt, generate_answer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

embeddings_dir = "catalog_embeddings"
catalog_dir = "catalog_data"
chunks_dir = "catalog_chunks"

TOP_K = 6
PDF_URL = "https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(token=hf_token)

# 1) Ingestion
def ingestion():
    pdf_bytes = download_pdf(PDF_URL)
    pages = extract_pages(pdf_bytes)
    save_pages(pages)
    print(f"\nDone — {len(pages)} pages extracted")



# 2) Chunking
def chunking():
    model = SentenceTransformer(EMBEDDING_MODEL)
    pages = load_pages(catalog_dir)
    all_chunks = chunk_catalog(pages, model)

    # Save combined chunks
    output_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")

# 3) Embedding
def embedding():
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = load_all_chunks(chunks_dir)
    embeddings = embed_chunks(chunks, model)

    embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    metadata_path = os.path.join(embeddings_dir, "chunks_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata   → {metadata_path}")


def main():
    """Main pipeline: ingestion → chunking → embedding → retrieval → generation"""
    
    # Step 1: Ingestion (skip if pages already exist)
    pages_dir = os.path.join(catalog_dir, "pages")
    if not os.path.exists(pages_dir) or len(os.listdir(pages_dir)) == 0:
        print("\n=== Step 1: Data Ingestion ===")
        ingestion()
    else:
        print("\n[✓ Skipping ingestion - pages already exist]")

    # Step 2: Chunking (skip if chunks already exist)
    chunks_file = os.path.join(chunks_dir, "_all_chunks.json")
    if not os.path.exists(chunks_file):
        print("\n=== Step 2: Chunking ===")
        chunking()
    else:
        print("\n[✓ Skipping chunking - chunks already exist]")

    # Step 3: Embedding (skip if embeddings already exist)
    embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
    if not os.path.exists(embeddings_file):
        print("\n=== Step 3: Embedding ===")
        embedding()
    else:
        print("\n[✓ Skipping embedding - embeddings already exist]")

    # Step 4 & 5: Retrieval + Generation
    print("\n=== Step 4 & 5: Retrieval + Generation ===")
    print("\nLoading retrieval system...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks = load_embeddings(embeddings_dir)
    print(f"Ready — {len(chunks)} chunks loaded\n")

    while True:
        query = input("Your question (or 'quit'): ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break

        print("\nRetrieving relevant chunks...")
        top_chunks = retrieve(query, model, embeddings, chunks, top_k=TOP_K)

        print("Generating answer...\n")
        answer = generate_answer(query, client, top_chunks)

        print("=" * 60)
        print(answer)
        print("=" * 60)

        print("\nSources used:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"  [{i}] Page {chunk['page_number']} (score: {chunk['score']})")
        print()


if __name__ == "__main__":
    main()


