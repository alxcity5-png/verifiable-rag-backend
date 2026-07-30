from app.services.llm_service import generate_text


CLAIM_EXTRACTION_PROMPT = """
You are an information extraction assistant.

Your task is to convert the answer into individual complete factual claims.

Use the question to understand the context of the answer.

Rules:
- Return one complete factual claim per line.
- The claim must be understandable without seeing the question.
- If the answer is a date, name, number, or short phrase, include the context from the question.
- Keep related details together in the same claim.
- Do not split dates, names, numbers, locations, or other details that belong to the same fact.
- Do not create duplicate claims.
- Do not add information that is not present in the question or answer.
- Preserve the original meaning.
- Do not number the claims.
- Do not use bullet points.
- If the answer contains no factual claims, return exactly:
NO_CLAIMS

Question:
{question}

Answer:
{answer}
"""


def extract_claims(question: str, answer: str) -> list[str]:
    """
    Extract individual factual claims from an answer.

    Args:
        question: Original user question.
        answer: Generated answer.

    Returns:
        List[str]: List of factual claims.
    """

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        question=question,
        answer=answer
    )

    response = generate_text(prompt)

    response = response.strip()

    if response == "NO_CLAIMS":
        return []

    claims = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    return claims