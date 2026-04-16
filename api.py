# api.py — FastAPI wrapper for the RAG pipeline

from fastapi import FastAPI, HTTPException
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

model = None
embeddings = None
chunks = None
client = None


def load_rag_system():
    global model, embeddings, chunks, client

    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    if embeddings is None or chunks is None:
        embeddings, chunks = load_embeddings(embeddings_dir)

    if client is None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set in the environment.")
        client = InferenceClient(token=hf_token)

    return model, embeddings, chunks, client


@app.on_event("startup")
def startup_event():
    try:
        load_rag_system()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize RAG system: {exc}") from exc


class Query(BaseModel):
    question: str


@app.get("/")
def health_check():
    if chunks is None:
        return {"status": "starting", "chunks_loaded": 0}
    return {"status": "running", "chunks_loaded": len(chunks)}


@app.post("/ask")
def ask(query: Query):
    try:
        current_model, current_embeddings, current_chunks, current_client = load_rag_system()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    top_chunks = retrieve(query.question, current_model, current_embeddings, current_chunks, top_k=TOP_K)
    answer = generate_answer(query.question, current_client, top_chunks)

    return {
        "question": query.question,
        "answer": answer,
        "sources": [
            {"page": c["page_number"], "score": c["score"]}
            for c in top_chunks
        ]
    }