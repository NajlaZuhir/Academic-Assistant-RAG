# 3) splits JSON policy text using semantic chunking with embeddings

import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

json_dir = "policy_jsons"
chunks_dir = "policy_chunks"
os.makedirs(chunks_dir, exist_ok=True)

CHUNK_SIZE = 500        # target characters per chunk
CHUNK_OVERLAP = 100     # character overlap
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.5  # merge chunks if similarity > threshold


def extract_text_from_json(json_data: dict) -> str:
    """Extract and combine text from JSON policy structure"""
    texts = []
    
    # Add headings
    if "headings" in json_data.get("content", {}):
        texts.extend(json_data["content"]["headings"])
    
    # Add paragraphs
    if "paragraphs" in json_data.get("content", {}):
        texts.extend(json_data["content"]["paragraphs"])
    
    # Add list items
    if "lists" in json_data.get("content", {}):
        for list_items in json_data["content"]["lists"]:
            texts.extend(list_items)
    
    # Add table content
    if "tables" in json_data.get("content", {}):
        for table in json_data["content"]["tables"]:
            for row in table:
                texts.extend(row)
    
    return "\n".join(texts)


def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def split_into_chunks_langchain(text: str) -> list[str]:
    """Split text using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def merge_semantic_chunks(chunks: list[str], model: SentenceTransformer, similarity_threshold: float = SIMILARITY_THRESHOLD) -> list[str]:
    """Merge chunks that are semantically similar."""
    if len(chunks) <= 1:
        return chunks
    
    print(f"    Embedding {len(chunks)} chunks for semantic analysis...")
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    merged_chunks = [chunks[0]]
    
    for i in range(1, len(chunks)):
        # Calculate similarity between current chunk and last merged chunk
        current_embedding = embeddings[i]
        last_embedding = embeddings[len(merged_chunks) - 1]
        
        # Cosine similarity
        similarity = np.dot(current_embedding, last_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(last_embedding) + 1e-10
        )
        
        if similarity > similarity_threshold:
            # Merge with previous chunk
            merged_chunks[-1] += " " + chunks[i]
        else:
            # Keep as separate chunk
            merged_chunks.append(chunks[i])
    
    print(f"    Semantic merge: {len(chunks)} → {len(merged_chunks)} chunks")
    return merged_chunks


def chunk_policy(json_path: str, policy_name: str, json_data: dict, model: SentenceTransformer) -> list[dict]:
    """Extract text from JSON and return semantically chunked dicts."""

    print(f"  Chunking: {policy_name}")

    raw_text = extract_text_from_json(json_data)
    text = clean_text(raw_text)
    
    # Initial split using LangChain
    initial_chunks = split_into_chunks_langchain(text)
    
    # Merge semantically similar chunks
    final_chunks = merge_semantic_chunks(initial_chunks, model)

    # Attach metadata to each chunk — important for retrieval later
    return [
        {
            "chunk_id": f"{policy_name}__chunk_{i}",
            "policy_name": policy_name,
            "source": json_path,
            "chunk_index": i,
            "total_chunks": len(final_chunks),
            "text": chunk,
        }
        for i, chunk in enumerate(final_chunks)
    ]


if __name__ == "__main__":
    print("Loading embedding model for semantic chunking...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    all_chunks = []

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in '{json_dir}'\n")

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        policy_name = json_file.replace(".json", "").replace("_", " ")
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        chunks = chunk_policy(json_path, policy_name, json_data, model) # chunk one policy with semantic merging
        all_chunks.extend(chunks) # add to master list

        # Save per-policy chunks as JSON
        chunk_path = os.path.join(chunks_dir, json_file.replace(".json", ".json"))
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        print(f"  → {len(chunks)} chunks saved to {chunk_path}\n")

    # Also save one combined file for easy loading later
    combined_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total Policies: {len(json_files)}")
    print(f"Total chunks across all policies: {len(all_chunks)}")
    print(f"Combined file saved to: {combined_path}")