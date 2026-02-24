
# UDST Academic Assistant

This repository contains a small Retrieval-Augmented Generation (RAG) pipeline and a Streamlit frontend for answering questions from the University of Doha for Science and Technology (UDST) Academic Catalog (2022–2023).

Contents
- [main.py](main.py): CLI pipeline that runs ingestion → chunking → embedding → retrieval → generation.
- [app.py](app.py): Streamlit web UI that queries the embeddings and displays answers with source citations.
- [requirements.txt](requirements.txt): Python dependencies required to run the project.
- `src/` — core pipeline modules:
	- [src/ingestion.py](src/ingestion.py): Download the catalog PDF and extract page-level text.
	- [src/chunking.py](src/chunking.py): Clean and split pages into semantically coherent chunks.
	- [src/embedding.py](src/embedding.py): Embed chunks into vectors and save to disk.
	- [src/retrieval.py](src/retrieval.py): Load embeddings and perform semantic search.
	- [src/generation.py](src/generation.py): Build prompt and call the LLM to generate answers.

Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv env
source env/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the repo root and add your Hugging Face token:

```
HF_TOKEN=your_hf_token_here
```

Generating the index (one-time)

If you don't already have precomputed chunks/embeddings in `catalog_chunks/` and `catalog_embeddings/`, run the full pipeline which will:
- Download the catalog PDF
- Extract pages
- Chunk pages
- Compute embeddings

```bash
python main.py
```

This will create:
- `catalog_data/catalog_pages.json`
- `catalog_chunks/_all_chunks.json`
- `catalog_embeddings/embeddings.npy`
- `catalog_embeddings/chunks_metadata.json`

Run the Streamlit app

Start the web UI:

```bash
streamlit run app.py
```

Usage notes
- The assistant answers using only excerpts from the UDST Academic Catalog (2022–2023) and lists source page numbers for traceability.
- To re-run only specific steps use the helper scripts in `src/`:
	- Extract pages: run `python src/ingestion.py`
	- Chunk pages: run `python src/chunking.py`
	- Create embeddings: run `python src/embedding.py`

Environment and configuration
- The catalog URL is configured in `src/ingestion.py` (PDF_URL) and in `main.py` as `PDF_URL` — update if you need a different PDF.
- The LLM client uses the `HF_TOKEN` environment variable. Ensure it is present in `.env` or exported in your shell.

Troubleshooting
- If the Streamlit app fails to load embeddings, run `python main.py` to regenerate them and ensure `catalog_embeddings/` contains `embeddings.npy` and `chunks_metadata.json`.
- If you get import errors, make sure your virtual environment is activated and dependencies from [requirements.txt](requirements.txt) are installed.

License & attribution
- This repository is a personal project sample built for UDST catalog retrieval. The catalog PDF is the authoritative source: https://www.udst.edu.qa/sites/default/files/2023-01/AcademicCatalog2022-2023.pdf

Contact
- For questions about this code or to request improvements, open an issue or contact the repository owner.

