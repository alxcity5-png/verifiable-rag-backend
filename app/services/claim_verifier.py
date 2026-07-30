from app.services.llm_service import generate_text


CLAIM_VERIFICATION_PROMPT = """
You are a fact verification assistant.

Your task is to verify whether a claim is supported by the provided evidence.

Rules:
- Only use the provided evidence.
- Do not use outside knowledge.
- If the evidence directly supports the claim, return SUPPORTED.
- If the evidence does not support the claim, return UNSUPPORTED.
- Return only one word.

Claim:
{claim}

Evidence:
{evidence}
"""


def verify_claim(claim: str, evidence_chunks: list[str]) -> dict:
    """
    Verify a claim against retrieved evidence.

    Returns:
        Dictionary containing verification result.
    """

    evidence = "\n\n".join(evidence_chunks)

    prompt = CLAIM_VERIFICATION_PROMPT.format(
        claim=claim,
        evidence=evidence
    )

    response = generate_text(prompt)

    supported = response.strip().upper() == "SUPPORTED"

    return {
        "claim": claim,
        "supported": supported,
        "evidence": evidence if supported else None
    }