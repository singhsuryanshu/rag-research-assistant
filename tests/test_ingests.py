"""
Run this to verify ingestion works:
    python tests/test_ingest.py
"""
import sys
sys.path.insert(0, ".")  # Allow imports from project root

from app.ingest import ingest_document, list_ingested_documents, get_vectorstore


def test_ingestion():
    # 1. Ingest a test file (create a small test PDF first)
    # For now, let's create a simple text file to test with
    with open("data/documents/test.txt", "w") as f:
        f.write("""
        Artificial intelligence (AI) is intelligence demonstrated by machines.
        Machine learning is a subset of AI that learns from data.
        Deep learning uses neural networks with many layers.
        Large language models (LLMs) are trained on massive text datasets.
        RAG stands for Retrieval-Augmented Generation.
        """ * 20)  # Repeat to get enough content to chunk

    # 2. Run ingestion
    result = ingest_document("data/documents/test.txt")
    print(f"Ingestion result: {result}")
    assert result["status"] == "success"
    assert result["chunks"] > 0

    # 3. Verify it's in the vector store
    docs = list_ingested_documents()
    print(f"Ingested docs: {docs}")
    assert "test.txt" in docs

    # 4. Do a quick similarity search to verify retrieval works
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search("What is machine learning?", k=3)
    print(f"Search results ({len(results)} found):")
    for r in results:
        print(f"  [{r.metadata['source']}] {r.page_content[:80]}...")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_ingestion()