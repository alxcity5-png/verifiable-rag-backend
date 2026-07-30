from app.services.llm_service import generate_text


CLAIM_EXTRACTION_PROMPT = """
You are a fact extraction assistant.

Your task is to extract only the most important factual claims from the answer.

Use the question to understand the context.

Rules:
- Return only complete standalone factual claims.
- Prefer fewer, stronger claims over many small claims.
- Combine related facts into one claim when they describe the same event or fact.
- Do not split names, dates, courses, locations, or numbers away from the fact they belong to.
- Do not create duplicate or overlapping claims.
- Do not repeat the same fact in different wording.
- Do not add information that is not present in the answer.
- Preserve the meaning of the answer.
- Do not number claims.
- Do not use bullet points.
- If there are no factual claims, return exactly:
NO_CLAIMS

Question:
{question}

Answer:
{answer}
"""


def extract_claims(question: str, answer: str) -> list[str]:
    """
    Extract high-quality factual claims from an answer.

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