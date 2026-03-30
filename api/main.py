from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="RAG Research Assistant",
    description="Upload documents and ask questions about them",
    version="1.0.0",
)

# CORS allows the Streamlit frontend (running on port 8501)
# to call the FastAPI backend (running on port 8000)
# In production, replace "*" with your actual domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "running", "message": "RAG API is live"}