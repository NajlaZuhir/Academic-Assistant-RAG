
# UDST Academic Assistant

## Overview
The **UDST Academic Assistant is a chatbot** that answers questions using the University of Doha for Science and Technology (UDST) [Academic Catalog](https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf). It uses a **retrieval-augmented generation (RAG) approach**: the system retrieves relevant catalog excerpts and generates concise, citation-backed answers.

## Models used
- Embedding model: BAAI/bge-small-en-v1.5
- Language model: meta-llama/Llama-3.1-8B-Instruct

## Tech Stack
- Python 3.10
- Streamlit

## Project Structure
```
├── src/
│   ├── ingestion.py      # Download the catalog PDF and extract page-level text
│   ├── chunking.py       # Clean and split pages into semantically coherent chunks
│   ├── embedding.py      # Embed chunks into vectors and save to disk
│   ├── retrieval.py      # Load embeddings and perform semantic search
│   └── generation.py     # Build prompt and call the LLM to generate answers
├── main.py               # CLI pipeline: ingestion → chunking → embedding → retrieval → generation
├── app.py                # Streamlit UI
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a `.env` file in the repo root and add your Hugging Face token:
# HF_TOKEN=your_hf_token_here

# Run the web UI
streamlit run app.py
```

**LLM RAG Project | 2025**
