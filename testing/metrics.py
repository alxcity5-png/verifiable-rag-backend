"""
metrics.py

Pure computation functions for benchmarking the Verifiable RAG backend.
No network calls here — run_benchmark.py handles hitting the API and
passes the raw responses in.

Design notes / honest limitations:
- The backend's /query/ endpoint always runs the full verification
  pipeline (there's no separate "naive mode" flag). So "Naive RAG" and
  "Verifiable RAG" are NOT two different API calls — they're two ways
  of *using* the same response:
    - Naive  = show `answer` to the user no matter what, ignore claims/
               verification entirely.
    - Verifiable = only show `answer` if faithfulness_score clears a
               threshold; otherwise warn the user / withhold it.
  This mirrors the retry/warn logic Prachi's pipeline already
  implements client-side of the LLM call.

- Answer correctness (used for Precision) is scored automatically via
  fuzzy text similarity against the ground_truth_answer, since we don't
  have a human grader in the loop. This is a heuristic, not ground
  truth — treat Precision numbers as directionally useful, not exact.

- Recall@K matches the ground-truth source chunk against retrieved
  chunks using fuzzy overlap, not exact string equality, because the
  backend's chunker doesn't guarantee a chunk boundary lines up with a
  full sentence (this improves after the sentence-aware chunking fix,
  but fuzzy matching keeps the benchmark robust either way).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import mean
from typing import Optional


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s%.]", " ", text)   # keep % and . (money/decimals)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity_ratio(a: str, b: str) -> float:
    """0.0-1.0 fuzzy similarity between two strings."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def extract_numbers(text: str) -> set[str]:
    """Pull out numeric tokens (fines, days, percentages, Rs. amounts)
    since these are the highest-signal facts in this dataset's domains."""
    return set(NUMBER_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def is_chunk_match(
    retrieved_chunk_text: str,
    correct_source_chunk: str,
    min_similarity: float = 0.35,
) -> bool:
    """
    A retrieved chunk 'counts' toward recall if it substantially
    overlaps with the ground-truth source chunk. Since the backend's
    stored chunks may be shorter fragments of the full ground-truth
    sentence(s) (fixed-size chunking) or exact sentence matches
    (sentence-aware chunking), we check similarity in both directions:
    either text is largely contained in the other.
    """
    a = normalize(retrieved_chunk_text)
    b = normalize(correct_source_chunk)

    if not a or not b:
        return False

    # Direct substring containment (common after the sentence-aware fix)
    if a in b or b in a:
        return True

    # Fuzzy fallback for partial/fixed-size chunk overlap
    return similarity_ratio(a, b) >= min_similarity


def recall_at_k(
    retrieved_sources: list[dict],
    correct_source_chunk: str,
    expected_source_document: Optional[str] = None,
) -> bool:
    """
    Returns True if the correct source chunk was found anywhere in the
    top-K retrieved sources for this question.

    If `expected_source_document` is provided and the retrieved chunk's
    metadata includes a "source" field (post-metadata-fix backend),
    also requires the retrieved chunk to come from the right document.
    Falls back to text-only matching if metadata isn't present yet.
    """
    for src in retrieved_sources:
        chunk_text = src.get("chunk", "")
        metadata = src.get("metadata") or {}
        source_doc = metadata.get("source")

        if expected_source_document and source_doc:
            if source_doc != expected_source_document:
                continue

        if is_chunk_match(chunk_text, correct_source_chunk):
            return True

    return False


# ---------------------------------------------------------------------------
# Answer correctness (feeds into Precision)
# ---------------------------------------------------------------------------

def score_answer_correctness(model_answer: str, ground_truth_answer: str) -> dict:
    """
    Heuristic correctness score for an answer, combining:
    - fuzzy text similarity (catches paraphrased-but-correct answers)
    - numeric overlap (catches the case where wording differs but the
      actual fine/deadline/percentage is right or wrong, which matters
      more than prose similarity for this dataset's domains)

    Returns both the continuous scores and a binary `is_correct` call
    at a threshold tuned to be reasonably strict without punishing
    minor paraphrasing.
    """
    if not model_answer or "could not find this information" in model_answer.lower():
        return {
            "text_similarity": 0.0,
            "number_overlap": 0.0,
            "is_correct": False,
        }

    text_sim = similarity_ratio(model_answer, ground_truth_answer)

    gt_numbers = extract_numbers(ground_truth_answer)
    ans_numbers = extract_numbers(model_answer)

    if gt_numbers:
        number_overlap = len(gt_numbers & ans_numbers) / len(gt_numbers)
    else:
        # No numbers in ground truth (rare in this dataset) — fall back
        # to text similarity only.
        number_overlap = text_sim

    # Weight numeric correctness higher: getting the Rs. amount or the
    # day count wrong is a worse failure than awkward phrasing.
    combined = (0.4 * text_sim) + (0.6 * number_overlap)

    return {
        "text_similarity": round(text_sim, 3),
        "number_overlap": round(number_overlap, 3),
        "is_correct": combined >= 0.55,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(per_question_results: list[dict]) -> dict:
    """
    Takes the list of per-question result dicts (see run_benchmark.py
    for the schema) and computes overall + per-domain summary stats
    for both the Naive and Verifiable views.
    """
    if not per_question_results:
        return {}

    def safe_mean(values):
        values = [v for v in values if v is not None]
        return round(mean(values), 3) if values else None

    overall = {
        "n_questions": len(per_question_results),
        "recall_at_k": safe_mean([r["recall_at_k"] for r in per_question_results]),
        "faithfulness": safe_mean([r["faithfulness_score"] for r in per_question_results]),
        "naive_precision": safe_mean(
            [1.0 if r["naive_is_correct"] else 0.0 for r in per_question_results]
        ),
        "verifiable_precision": safe_mean(
            [1.0 if r["verifiable_is_correct"] else 0.0 for r in per_question_results if r["verifiable_shown"]]
        ),
        "verifiable_coverage": safe_mean(
            [1.0 if r["verifiable_shown"] else 0.0 for r in per_question_results]
        ),
        "avg_latency_ms": safe_mean([r["latency_ms"] for r in per_question_results]),
    }

    domains = {}
    for r in per_question_results:
        domains.setdefault(r["domain"], []).append(r)

    per_domain = {}
    for domain, results in domains.items():
        per_domain[domain] = {
            "n_questions": len(results),
            "recall_at_k": safe_mean([r["recall_at_k"] for r in results]),
            "faithfulness": safe_mean([r["faithfulness_score"] for r in results]),
            "naive_precision": safe_mean(
                [1.0 if r["naive_is_correct"] else 0.0 for r in results]
            ),
            "verifiable_precision": safe_mean(
                [1.0 if r["verifiable_is_correct"] else 0.0 for r in results if r["verifiable_shown"]]
            ),
            "verifiable_coverage": safe_mean(
                [1.0 if r["verifiable_shown"] else 0.0 for r in results]
            ),
            "avg_latency_ms": safe_mean([r["latency_ms"] for r in results]),
        }

    return {"overall": overall, "per_domain": per_domain}