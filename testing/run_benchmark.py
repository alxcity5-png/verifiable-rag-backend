"""
run_benchmark.py

Runs the full test_dataset.json (120 questions) through the Verifiable
RAG backend's POST /query/ endpoint and produces Naive-vs-Verifiable
comparison metrics.

IMPORTANT: the backend only has ONE query mode — it always runs claim
extraction + verification. So this script makes exactly ONE API call
per question, and derives both the "naive" and "verifiable" view from
that single response:
    - Naive view:       always shows `answer` to the user as-is.
    - Verifiable view:  only shows `answer` if faithfulness_score >=
                         --faithfulness-threshold; otherwise the
                         answer is withheld and the user is warned.

Usage:
    python run_benchmark.py \\
        --api-url http://localhost:8000 \\
        --dataset test_dataset.json \\
        --output-dir ./reports \\
        --top-k 3 \\
        --faithfulness-threshold 0.7

Requires:
    pip install requests
"""

import argparse
import json
import time
from pathlib import Path

import requests

from metrics import (
    recall_at_k,
    score_answer_correctness,
    aggregate_metrics,
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3
REQUEST_TIMEOUT = 60


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def call_query_endpoint(api_url: str, question: str, top_k: int) -> dict | None:
    """
    Calls POST {api_url}/query/ with retries. Returns the parsed JSON
    response, or None if all retries were exhausted (the question is
    logged as failed and skipped rather than crashing the whole run).
    """
    url = api_url.rstrip("/") + "/query/"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json={"question": question, "top_k": top_k},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 200:
                return response.json()

            print(f"  [warn] HTTP {response.status_code} on attempt {attempt}/{MAX_RETRIES}: {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            print(f"  [warn] request error on attempt {attempt}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"  [error] giving up after {MAX_RETRIES} attempts")
    return None


def run_single_question(
    api_url: str,
    item: dict,
    top_k: int,
    faithfulness_threshold: float,
) -> dict | None:
    """Runs one test-dataset question through the backend and packages
    all the raw + derived data needed for both metrics and debugging."""

    start = time.perf_counter()
    response = call_query_endpoint(api_url, item["question"], top_k)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    if response is None:
        return {
            "id": item["id"],
            "domain": item["domain"],
            "question": item["question"],
            "error": "request_failed",
            "latency_ms": latency_ms,
            "recall_at_k": None,
            "faithfulness_score": None,
            "naive_is_correct": False,
            "verifiable_shown": False,
            "verifiable_is_correct": False,
        }

    model_answer = response.get("answer", "")
    faithfulness_score = response.get("faithfulness_score", 0.0)
    sources = response.get("sources", [])
    claims = response.get("claims", [])
    verification = response.get("verification", [])

    # --- Recall@K ---
    recall_hit = recall_at_k(
        retrieved_sources=sources,
        correct_source_chunk=item["correct_source_chunk"],
        expected_source_document=item.get("source_document"),
    )

    # --- Naive view: always shown, scored against ground truth ---
    naive_score = score_answer_correctness(model_answer, item["ground_truth_answer"])

    # --- Verifiable view: only "shown" if faithfulness clears threshold ---
    verifiable_shown = faithfulness_score >= faithfulness_threshold
    verifiable_score = naive_score if verifiable_shown else {
        "text_similarity": None, "number_overlap": None, "is_correct": False
    }

    return {
        "id": item["id"],
        "domain": item["domain"],
        "question": item["question"],
        "ground_truth_answer": item["ground_truth_answer"],
        "model_answer": model_answer,
        "claims": claims,
        "verification": verification,
        "faithfulness_score": faithfulness_score,
        "recall_at_k": recall_hit,
        "latency_ms": latency_ms,
        "naive_is_correct": naive_score["is_correct"],
        "naive_text_similarity": naive_score["text_similarity"],
        "naive_number_overlap": naive_score["number_overlap"],
        "verifiable_shown": verifiable_shown,
        "verifiable_is_correct": verifiable_score["is_correct"],
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Verifiable RAG backend.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the FastAPI backend")
    parser.add_argument("--dataset", default="test_dataset.json", help="Path to test_dataset.json")
    parser.add_argument("--output-dir", default="./reports", help="Where to write result JSON files")
    parser.add_argument("--top-k", type=int, default=3, help="top_k passed to /query/")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.7,
                         help="Minimum faithfulness_score for the Verifiable view to show an answer")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick testing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Running {len(dataset)} questions against {args.api_url} (top_k={args.top_k})...\n")

    results = []
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {item['id']} ({item['domain']})")
        result = run_single_question(
            api_url=args.api_url,
            item=item,
            top_k=args.top_k,
            faithfulness_threshold=args.faithfulness_threshold,
        )
        results.append(result)

    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"\n{len(failed)} question(s) failed after retries: {[r['id'] for r in failed]}")

    successful_results = [r for r in results if not r.get("error")]

    # --- Save raw results, split by view for downstream chart scripts ---
    naive_results = [
        {
            "id": r["id"], "domain": r["domain"], "question": r["question"],
            "answer": r["model_answer"], "is_correct": r["naive_is_correct"],
            "text_similarity": r["naive_text_similarity"],
            "number_overlap": r["naive_number_overlap"],
            "latency_ms": r["latency_ms"],
        }
        for r in successful_results
    ]
    verifiable_results = [
        {
            "id": r["id"], "domain": r["domain"], "question": r["question"],
            "answer": r["model_answer"] if r["verifiable_shown"] else None,
            "shown": r["verifiable_shown"], "is_correct": r["verifiable_is_correct"],
            "faithfulness_score": r["faithfulness_score"],
            "claims": r["claims"], "verification": r["verification"],
            "recall_at_k": r["recall_at_k"],
            "latency_ms": r["latency_ms"],
        }
        for r in successful_results
    ]

    with open(output_dir / "naive_results.json", "w") as f:
        json.dump(naive_results, f, indent=2)
    with open(output_dir / "verifiable_results.json", "w") as f:
        json.dump(verifiable_results, f, indent=2)
    with open(output_dir / "raw_results_full.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Aggregate + print summary table ---
    summary = aggregate_metrics(successful_results)
    with open(output_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    for k, v in summary.get("overall", {}).items():
        print(f"  {k:25s}: {v}")

    print("\n" + "=" * 70)
    print("PER-DOMAIN SUMMARY")
    print("=" * 70)
    for domain, stats in summary.get("per_domain", {}).items():
        print(f"\n{domain}")
        for k, v in stats.items():
            print(f"  {k:25s}: {v}")

    print(f"\nResults written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()