def calculate_faithfulness(verification_results: list[dict]) -> float:
    """
    Calculate faithfulness score based on claim verification.

    Score:
        supported claims / total claims
    """

    if not verification_results:
        return 0.0

    supported_claims = sum(
        1
        for result in verification_results
        if result.get("supported") is True
    )

    total_claims = len(verification_results)

    return round(
        supported_claims / total_claims,
        2
    )