from app.services.llm_service import generate_text


CLAIM_VERIFICATION_PROMPT = """
You are a fact verification assistant.

Your task is to verify whether a claim is supported by the provided evidence.

Rules:
- Only use the provided evidence.
- Do not use outside knowledge.
- If the evidence supports the claim, return:
SUPPORTED
Evidence: <exact supporting sentence from the evidence>

- If the evidence does not support the claim, return exactly:
UNSUPPORTED

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

    response = generate_text(prompt).strip()

    if response.startswith("SUPPORTED"):
        evidence_text = response.replace(
            "SUPPORTED",
            "",
            1
        ).replace(
            "Evidence:",
            "",
            1
        ).strip()

        return {
            "claim": claim,
            "supported": True,
            "evidence": evidence_text
        }

    return {
        "claim": claim,
        "supported": False,
        "evidence": None
    }