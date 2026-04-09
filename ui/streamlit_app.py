import os
import sys
sys.path.insert(0, ".")

import streamlit as st
import requests
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────
# Must be the FIRST Streamlit command in the file
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
)

# Backend API (local default). Override for Streamlit Cloud:
# - App Settings → Secrets: API_BASE = "https://your-backend.example.com/api"
# - or env: API_BASE=https://your-backend.example.com/api
_default_api = "http://127.0.0.1:8000/api"
API_BASE = (os.environ.get("API_BASE") or st.secrets.get("API_BASE", _default_api)).rstrip(
    "/"
)

# ── Session state initialization ────────────────────────────────────
# st.session_state persists across reruns within the same browser session
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "documents" not in st.session_state:
    st.session_state.documents = []

# ── Helper functions ────────────────────────────────────────────────

def fetch_documents():
    """Fetch list of ingested documents from API."""
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        return resp.json().get("documents", [])
    except:
        return []

def upload_file(file) -> dict:
    """Upload a file to the API and return ingestion result."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def ask_question(question: str, filter_source: str = None) -> dict:
    """Send a question to the API and return the answer."""
    payload = {"question": question}
    if filter_source:
        payload["filter_source"] = filter_source
    resp = requests.post(f"{API_BASE}/query", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Research Assistant")
    st.divider()

    # File upload section
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF, Word (.docx), Text (.txt)"
    )

    if uploaded is not None:
        if st.button("Ingest Document", type="primary"):
            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    result = upload_file(uploaded)
                    st.success(
                        f"✓ Ingested {result['chunks']} chunks "
                        f"from {result['pages']} pages"
                    )
                    # Refresh document list
                    st.session_state.documents = fetch_documents()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()

    # Document library
    st.subheader("Ingested Documents")
    st.session_state.documents = fetch_documents()

    selected_doc = None
    if st.session_state.documents:
        # "All documents" + each individual doc
        options = ["All documents"] + st.session_state.documents
        selected = st.selectbox("Search in:", options)
        selected_doc = None if selected == "All documents" else selected

        for doc in st.session_state.documents:
            st.text(f"📄 {doc}")
    else:
        st.info("No documents ingested yet")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ───────────────────────────────────────────────────
st.header("Ask your documents")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 Sources ({len(msg['sources'])} chunks)"):
                for source in msg["sources"]:
                    st.markdown(
                        f"**{source['source']}** — page {source['page']}"
                    )
                    st.caption(source["excerpt"])
                    st.divider()

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = ask_question(question, filter_source=selected_doc)
                st.markdown(result["answer"])

                # Show sources in expander
                if result["sources"]:
                    with st.expander(
                        f"📎 Sources ({len(result['sources'])} chunks)"
                    ):
                        for source in result["sources"]:
                            st.markdown(
                                f"**{source['source']}** — page {source['page']}"
                            )
                            st.caption(source["excerpt"])
                            st.divider()

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                st.error(f"Error: {e}. Is the API server running?")