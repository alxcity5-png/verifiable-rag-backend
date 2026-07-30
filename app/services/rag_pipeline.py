from app.services.retriever import retrieve_chunks
from app.services.llm_service import generate_answer


def answer_question(question: str, top_k: int = 3):
    """
    Complete RAG pipeline:
    Question -> Retrieval -> LLM answer
    """

    # Retrieve evidence
    chunks = retrieve_chunks(
        question=question,
        top_k=top_k
    )


    # Combine retrieved chunks into context
    context = "\n\n".join(
        [
            chunk["chunk"]
            for chunk in chunks
        ]
    )


    # Create prompt for LLM
    prompt = f"""
You are a helpful assistant answering questions from documents.

Use only the provided context.
If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""


    # Generate answer
    answer = generate_answer(prompt)


    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }