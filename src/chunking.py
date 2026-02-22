# 3) splits JSON policy text into overlapping chunks

import os
import re
import json

json_dir = "policy_jsons"
chunks_dir = "policy_chunks"
os.makedirs(chunks_dir, exist_ok=True)

CHUNK_SIZE = 500        # number of words per chunk
CHUNK_OVERLAP = 50      # words shared between consecutive chunks


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


def chunk_policy(json_path: str, policy_name: str, json_data: dict) -> list[dict]:
    """Extract text from JSON and return a list of chunk dicts."""

    print(f"  Chunking: {policy_name}")

    raw_text = extract_text_from_json(json_data)
    text = clean_text(raw_text)
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    # Attach metadata to each chunk — important for retrieval later
    return [
        {
            "chunk_id": f"{policy_name}__chunk_{i}",
            "policy_name": policy_name,
            "source": json_path,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    all_chunks = []

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files in '{json_dir}'\n")

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        policy_name = json_file.replace(".json", "").replace("_", " ")
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        chunks = chunk_policy(json_path, policy_name, json_data) # chunk one policy
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