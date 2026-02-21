
import os
import re
import json
from pypdf import PdfReader

txt_dir = "policy_txts"
chunks_dir = "policy_chunks"
os.makedirs(chunks_dir, exist_ok=True)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30


def load_text_from_file(txt_path: str) -> str:
    """Read clean text directly from saved .txt file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_policy(txt_path: str, policy_name: str) -> list[dict]:
    print(f"  Chunking: {policy_name}")
    text = load_text_from_file(txt_path)

    if not text.strip():
        print(f"  [WARN] Empty text for {policy_name}")
        return []

    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    return [
        {
            "chunk_id": f"{policy_name}__chunk_{i}",
            "policy_name": policy_name,
            "source": txt_path,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


if __name__ == "__main__":
    all_chunks = []
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} text files in '{txt_dir}'\n")

    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        policy_name = txt_file.replace(".txt", "").replace("_", " ")
        chunks = chunk_policy(txt_path, policy_name)
        all_chunks.extend(chunks)

        chunk_path = os.path.join(chunks_dir, txt_file.replace(".txt", ".json"))
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  → {len(chunks)} chunks saved\n")

    combined_path = os.path.join(chunks_dir, "_all_chunks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved → {combined_path}")