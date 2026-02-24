
# UDST Academic Assistant
## Overview
The UDST Academic Assistant Chatbot is designed to answer questions from the University of Doha for Science and Technology (UDST) Academic Catalog (2022–2023) [https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf]. It utilizes retrieval-augmented generation (RAG) to fetch relevant policy information and present structured responses.

For the models i used the following:

Embedding model: "BAAI/bge-small-en-v1.5"
Language model: "meta-llama/Llama-3.1-8B-Instruct"

## Tech Stack
- Python 3.10
- Streamlit

## 📁 Project Structure
```
├── src/
│   ├── ingestion.py      # Download the catalog PDF and extract page-level text.
|	├── chunking.py       # Clean and split pages into semantically coherent chunks.
|	├── embedding.py      # Embed chunks into vectors and save to disk.
|	├── retrieval.py      # Load embeddings and perform semantic search.
|	├── generation.py     # Build prompt and call the LLM to generate answers.
├── main.py               # CLI pipeline that runs ingestion → chunking → embedding → retrieval → generation.
├── app.py                # Streamlit UI
└── requirements.txt
```


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a `.env` file in the repo root and add your Hugging Face token
HF_TOKEN=your_hf_token_here

# Run web UI
streamlit run app.py
```

**LLM RAG Project | 2026**
