# 3) Loads PDF policies, extracts text, and chunks them for indexing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

POLICY_FOLDER = "policy_pdfs"
os.makedirs(POLICY_FOLDER, exist_ok=True)

def load_policies(folder):
    policy_content = []
    total_files = len([file for file in os.listdir(folder) ])
    print(f"Loading policies from '{folder}'... Total PDF files found: {total_files}")

    for file in os.listdir(folder):
        try:
            pdf_path = os.path.join(folder, file)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            if text.strip():  # Only add if text was extracted
                policy_content.append({
                    "filename": file,
                    "content": text
                })
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return policy_content


def chunk_documents():
    documents = load_policies(POLICY_FOLDER)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked_docs = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "source": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })

    return chunked_docs


if __name__ == "__main__":
    chunked_documents = chunk_documents()
    
    # Group chunks by source document
    docs_by_source = {}
    for chunk in chunked_documents:
        source = chunk["source"]
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(chunk)
    
    # Display available documents with indices
    doc_list = list(docs_by_source.items())
    print(f"\n{'='*80}")
    print("Available Documents:\n")
    for idx, (source, chunks) in enumerate(doc_list):
        print(f"{idx + 1}. {source}")
    
    # Get user input for index
    doc_index = int(input(f"\nSelect document by index (0-{len(doc_list)-1}): ").strip())
    
    if 0 <= doc_index < len(doc_list):
        source, chunks = doc_list[doc_index]
        total_words = sum(len(chunk["text"].split()) for chunk in chunks)
        print(f"\n{'='*80}")
        print(f"\n[PDF] {source}")
        print(f"   Full Policy Document ({total_words:,} words)")
        print(f"   Total chunks: {len(chunks)}\n")
        
        chunk_size_tokens = int(800 * 1.3)  # ~800 words = ~1040 tokens
        overlap_tokens = int(150 * 1.3)     # ~150 words overlap = ~195 tokens
        
        for i, chunk in enumerate(chunks, 1):
            words = len(chunk["text"].split())
            tokens = int(words * 1.3)
            
            # Calculate start and end with consideration for overlap
            if i == 1:
                start = 0
                end = tokens
            else:
                # Each chunk starts where previous chunk started + (chunk_size - overlap)
                start = (i - 1) * (chunk_size_tokens - overlap_tokens)
                end = start + tokens
            
            print(f"   Chunk {i} ({start}–{end} tokens)")
        
        print("-" * 80)
    else:
        print("Invalid index!")