import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingest import ingest_document, list_ingested_documents
from app.chain import answer_question
from app.utils import DOCUMENTS_PATH

router = APIRouter()


# ── Request / Response models ────────────────────────────────────
# Pydantic models define the shape of JSON bodies.
# FastAPI uses them to validate input and generate API docs.

class QueryRequest(BaseModel):
    question: str
    k: int = 5                    # How many chunks to retrieve
    filter_source: Optional[str] = None  # Restrict to one document


class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str

# ── Routes ───────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accept a file upload, save it to disk, and run ingestion.

    UploadFile is FastAPI's built-in type for multipart/form-data uploads.
    The file is received as a stream and saved to the documents directory.
    """
    # Validate file type
    allowed = {".pdf", ".docx", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not supported. Use: {allowed}"
        )

    # Save file to disk
    save_path = Path(DOCUMENTS_PATH) / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion pipeline
    try:
        result = ingest_document(str(save_path))
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Answer a question using the RAG pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = answer_question(
        question=request.question,
        k=request.k,
        filter_source=request.filter_source,
    )
    return result

@router.get("/documents")
async def get_documents():
    """List all ingested documents."""
    docs = list_ingested_documents()
    return {"documents": docs, "count": len(docs)}