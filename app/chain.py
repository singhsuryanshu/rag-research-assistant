from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from app.retriever import retrieve_relevant_chunks, format_context
from app.prompts import RAG_PROMPT
from app.utils import LLM_MODEL, OPENAI_API_KEY, RETRIEVAL_K

def get_llm() -> ChatOpenAI:
    """
    Initialize the language model.
    ChatOpenAI wraps OpenAI's chat completion endpoint.

    Model options and tradeoffs:
    - gpt-4o-mini: Fast, cheap (~$0.00015/1k input tokens). Good for most queries.
    - gpt-4o: Slower, more expensive (~$0.005/1k). Better reasoning.
    - gpt-3.5-turbo: Very cheap, older model, less capable.

    temperature=0 means deterministic output (no randomness).
    For a research assistant, you want consistent, factual answers.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

def answer_question(
    question: str,
    k: int = RETRIEVAL_K,
    filter_source: str = None,
) -> Dict[str, Any]:
    """
    The core RAG function. Given a question, returns an answer with sources.

    Returns:
        {
            "answer": str,           # The LLM's answer
            "sources": List[dict],   # List of chunks used
            "question": str,         # Echo of the input question
        }
    """
    # Step 1: Retrieve relevant chunks
    chunks: List[Document] = retrieve_relevant_chunks(
        query=question,
        k=k,
        filter_source=filter_source,
    )

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Please upload a document first.",
            "sources": [],
            "question": question,
        }

    # Step 2: Format chunks into context string
    context = format_context(chunks)

    # Step 3: Build the prompt by filling in the template
    prompt_text = RAG_PROMPT.format(
        context=context,
        question=question,
    )

    # Step 4: Call the LLM
    llm = get_llm()
    # invoke() sends the prompt and returns an AIMessage object
    response = llm.invoke(prompt_text)

    # Step 5: Extract and structure the output
    answer = response.content  # .content is the text of the LLM response

    # Format sources for the response
    sources = [
        {
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "excerpt": chunk.page_content[:200] + "...",
        }
        for chunk in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "question": question,
    }
