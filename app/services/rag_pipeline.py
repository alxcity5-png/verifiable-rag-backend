from app.services.retriever import retrieve_chunks
from app.services.llm_service import generate_answer


def clean_text(text: str) -> str:
    """
    Remove problematic control characters from extracted PDF text.
    """
    return (
        text.replace("\x00", "")
            .replace("\r", "")
            .strip()
    )


def answer_question(question: str, top_k: int = 3):
    chunks = retrieve_chunks(
        question=question,
        top_k=top_k
    )

    context = "\n\n".join(
        clean_text(chunk["chunk"])
        for chunk in chunks
    )

    answer = generate_answer(
        context=context,
        question=question
    )

    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }