from typing import List, Tuple
from langchain.schema import Document
from app.ingest import get_vectorstore
from app.utils import RETRIEVAL_K


def retrieve_relevant_chunks(
    query: str,
    k: int = RETRIEVAL_K,
    filter_source: str = None,
) -> List[Document]:
    """
    Search ChromaDB for the k most relevant chunks for a query.

    Args:
        query: The user's question (plain text)
        k: How many chunks to return (default 5)
        filter_source: Optional filename to restrict search to one document

    Returns:
        List of Document objects sorted by relevance (most relevant first)

    How it works:
        1. OpenAI embeds the query string into a vector
        2. ChromaDB computes cosine similarity between query vector
           and all stored chunk vectors
        3. Returns the k chunks with highest similarity scores
    """
    vectorstore = get_vectorstore()

    # Build optional filter (restrict to one document)
    where_filter = {"source": filter_source} if filter_source else None

    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter=where_filter,
    )

def retrieve_with_scores(
    query: str,
    k: int = RETRIEVAL_K,
) -> List[Tuple[Document, float]]:
    """
    Same as retrieve_relevant_chunks but also returns similarity scores.
    Useful for debugging — scores close to 1.0 = very relevant.
    Scores are cosine similarity values between 0 and 1.
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query=query, k=k)

def format_context(chunks: List[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Each chunk is formatted as:
        [filename, page N]
        ... chunk text ...

    This format makes it easy for the LLM to cite sources correctly.
    """
    context_parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        context_parts.append(
            f"[{source}, page {page}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

