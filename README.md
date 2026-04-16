
# 📚 UDST Academic Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that answers student questions 
using the University of Doha for Science and Technology (UDST) [Academic Catalog](https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf). 

## 💡 How It Works
```
PDF Catalog → Extract Pages → Chunk → Embed → Semantic Search → LLM Answer
```
1. **Ingestion** — Downloads the catalog PDF and extracts text page by page
2. **Chunking** — Cleans and splits pages into semantically coherent chunks
3. **Embedding** — Converts chunks into vectors using BGE and saves to disk
4. **Retrieval** — Finds the most relevant chunks for a user query via cosine similarity
5. **Generation** — Sends retrieved chunks + query to LLaMA to generate a grounded answer
   
## 🤖 Models

| Role | Model |
|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` |
| Language Model | `meta-llama/Llama-3.1-8B-Instruct` |


## 🛠️ Tech Stack

- **Python** 3.10
- **Streamlit** — conversational web UI
- **FastAPI** — REST API endpoint
- **LangChain** — recursive text splitting
- **sentence-transformers** — BGE embeddings
- **HuggingFace** InferenceClient — LLaMA inference
- **pypdf** — PDF extraction

  
## 📁 Project Structure
```
├── src/
│   ├── ingestion.py      # Download the catalog PDF and extract page-level text
│   ├── chunking.py       # Clean and split pages into semantically coherent chunks
│   ├── embedding.py      # Embed chunks into vectors and save to disk
│   ├── retrieval.py      # Load embeddings and perform semantic search
│   └── generation.py     # Build prompt and call the LLM to generate answers
├── main.py               # CLI pipeline: ingestion → chunking → embedding → retrieval → generation
├── app.py                # Streamlit  Chat UI
├── api.py                # FastAPI REST endpoint
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a `.env` file in the repo root and add your Hugging Face token:
HF_TOKEN=your_hf_token_here

# Run the pipeline (first time only — downloads PDF, chunks, embeds)
python main.py

# Run the web UI
streamlit run app.py
```

## 💬 Example Questions

- What are the admission requirements?
- What is the maximum number of allowed absences?
- What are the graduation requirements for a bachelor's degree?
- What scholarships are available for students?

---
**LLM RAG Project | 2025**
