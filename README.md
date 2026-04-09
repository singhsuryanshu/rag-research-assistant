# RAG Research Assistant

A lightweight Retrieval-Augmented Generation (RAG) app that lets you upload documents, index them in ChromaDB, and ask questions with grounded answers.

## What it does

- Ingests `pdf`, `docx`, and `txt` files
- Splits content into chunks and stores embeddings in ChromaDB
- Retrieves relevant context for each question
- Generates answers through an LLM via FastAPI
- Provides a simple Streamlit chat interface

## Environment setup (`.env`)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here

CHROMA_DB_PATH=./data/chroma_db
DOCUMENTS_PATH=./data/documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=5
LLM_MODEL=gpt-4o-mini
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```bash
streamlit run ui/streamlit_app.py
```
