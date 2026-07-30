from app.services.llm_service import generate_text


CLAIM_EXTRACTION_PROMPT = """
You are an information extraction assistant.

Your task is to split the provided answer into individual factual claims.

Rules:
- Return one complete factual claim per line.
- Keep related details together in the same claim.
- Do not split dates, names, numbers, locations, or other details that belong to the same fact.
- Do not create duplicate claims.
- Do not add information that is not present in the answer.
- Preserve the original meaning and wording as much as possible.
- Do not number the claims.
- Do not use bullet points.
- If the answer contains no factual claims, return exactly:
NO_CLAIMS

Answer:
{answer}
"""


def extract_claims(answer: str) -> list[str]:
    """
    Extract individual factual claims from an answer.

    Returns:
        List[str]: List of factual claims.
    """

    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)

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