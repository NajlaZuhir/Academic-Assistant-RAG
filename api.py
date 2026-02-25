# api.py — FastAPI wrapper for the RAG pipeline

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

from src.retrieval import load_embeddings, retrieve
from src.generation import generate_answer

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
embeddings_dir = "catalog_embeddings"
TOP_K = 6

app = FastAPI(title="UDST Academic Assistant API")

# Load once on startup — not on every request
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings, chunks = load_embeddings(embeddings_dir)
client = InferenceClient(token=os.getenv("HF_TOKEN"))


class Query(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "running", "chunks_loaded": len(chunks)}


@app.post("/ask")
def ask(query: Query):
    top_chunks = retrieve(query.question, model, embeddings, chunks, top_k=TOP_K)
    answer = generate_answer(query.question, client, top_chunks)

    return {
        "question": query.question,
        "answer": answer,
        "sources": [
            {"page": c["page_number"], "score": c["score"]}
            for c in top_chunks
        ]
    }