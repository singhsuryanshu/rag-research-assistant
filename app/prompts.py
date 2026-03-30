from langchain.prompts import PromptTemplate

# This is the main RAG prompt.
# {context} will be filled with retrieved document chunks.
# {question} will be filled with the user's question.
#
# Key design decisions:
# 1. "ONLY the provided context" → prevents hallucination
# 2. "If you don't know" → graceful fallback
# 3. "cite the source" → forces attribution
# 4. "Answer concisely but completely" → controls output length

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a research assistant answering questions based on uploaded documents.

Use ONLY the information provided in the context below to answer the question.
Do not use any outside knowledge. If the answer is not in the context, say:
"I don't have enough information in the uploaded documents to answer this."

For each piece of information you use, cite the source in this format: [filename, page N]

Context:
{context}

Question: {question}

Answer (be concise but complete, cite sources):"""
)


# A simpler prompt for when you want a one-sentence summary of each chunk
CHUNK_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize this text in one sentence:

{text}

Summary:"""
)