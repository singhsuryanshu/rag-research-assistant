rag-research-assistant/
├── app/ # Core logic
│ ├── ingest.py # Load → Chunk → Embed → Store
│ ├── retriever.py # Search ChromaDB by query
│ ├── chain.py # RAG chain: retrieve → prompt → LLM
│ ├── prompts.py # Prompt templates
│ └── utils.py # Helpers
├── api/ # FastAPI backend
│ ├── main.py # App entry point
│ └── routes.py # /upload /query /documents
├── ui/ # Frontend
│ └── streamlit_app.py # Streamlit chat UI
├── data/
│ ├── documents/ # Raw uploaded files
│ └── chroma_db/ # ChromaDB persistence dir
├── tests/
│ ├── test_ingest.py
│ └── test_retriever.py
├── .env # API keys (never commit)
├── .gitignore
├── requirements.txt
└── README.m