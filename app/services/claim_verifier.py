from app.services.llm_service import generate_text


CLAIM_VERIFICATION_PROMPT = """
You are a strict fact verification assistant.

Determine whether the claim is supported by the evidence.

Rules:
- Only use the provided evidence.
- Do not use outside knowledge.
- Judge meaning, not exact wording.
- Different wording with the same meaning should be considered supported.
- Legal terms such as "shall come into force", "will come into force", and "takes effect" should be treated as equivalent when referring to the same date.
- If names, dates, numbers, and facts match, mark the claim as SUPPORTED.

Output format:

If supported:
SUPPORTED
Evidence: <exact supporting sentence from evidence>

If not supported:
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

    evidence = "\n\n".join(evidence_chunks[:2])

    prompt = CLAIM_VERIFICATION_PROMPT.format(
        claim=claim,
        evidence=evidence
    )

    response = generate_text(prompt).strip()
    print("VERIFIER RESPONSE:")
    print(response)

    if response.upper().startswith("SUPPORTED"):
        evidence_text = (
            response.replace("SUPPORTED", "", 1)
            .replace("Evidence:", "", 1)
            .strip()
        )

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