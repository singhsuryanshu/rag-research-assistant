import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file into os.environ
# This must happen before any other import that uses env vars
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data/documents")

# ── Chunking ────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Retrieval ───────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# ── Model ───────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment. "
        "Make sure your .env file exists and contains OPENAI_API_KEY."
    )

# ── Ensure directories exist ────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)