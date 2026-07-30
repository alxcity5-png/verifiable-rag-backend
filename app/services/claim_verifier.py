from app.services.llm_service import generate_text


CLAIM_VERIFICATION_PROMPT = """
You are a strict evidence verification system.

Your job is to determine if the CLAIM is directly supported by the EVIDENCE.

Rules:
1. Use ONLY the provided evidence.
2. A claim is SUPPORTED if the evidence contains the same fact, even if the wording is different.
3. Do not require word-for-word matching.
4. Ignore missing extra details unless the claim contradicts the evidence.

Return:
SUPPORTED
Evidence: <exact supporting sentence>

OR

UNSUPPORTED

CLAIM:
{claim}

EVIDENCE:
{evidence}
"""


def verify_claim(claim: str, evidence_chunks: list[str]) -> dict:
    """
    Verify a claim against retrieved evidence.
    """

    evidence = "\n\n".join(evidence_chunks)

    prompt = CLAIM_VERIFICATION_PROMPT.format(
        claim=claim,
        evidence=evidence
    )

    response = generate_text(prompt).strip()

    print("VERIFIER RESPONSE:", response)

    if "SUPPORTED" in response.upper():
        evidence_text = response.split("Evidence:", 1)[-1].strip()

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