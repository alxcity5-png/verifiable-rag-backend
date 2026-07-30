"""
watch_and_ingest.py

Watches a folder for new PDF files and automatically triggers FastAPI's
document ingestion endpoint (chunk -> embed -> store) whenever one appears.

This is intentionally decoupled from run_benchmark.py / metrics.py:
- It only talks to the /documents endpoint, not /query or /eval.
- It can be developed, tested, and run independently of whether the
  benchmark script or Prachi's retrieval/verification logic is ready.

Usage:
    python watch_and_ingest.py --watch-dir ./incoming_pdfs --api-url http://localhost:8000

Requires:
    pip install watchdog requests
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_ENDPOINT = "/documents"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
# Some editors/browsers write files in chunks; wait for the file size to
# stop changing before treating it as "fully written".
STABLE_CHECK_INTERVAL = 1.0
STABLE_CHECK_ROUNDS = 3

PROCESSED_LOG = "processed_files.json"
PROCESSED_DIR_NAME = "_ingested"
FAILED_DIR_NAME = "_failed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("watch_and_ingest")


# ---------------------------------------------------------------------------
# Processed-file tracking (avoids re-ingesting the same file on restart)
# ---------------------------------------------------------------------------

def load_processed_log(watch_dir: Path) -> dict:
    log_path = watch_dir / PROCESSED_LOG
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s, starting fresh", log_path)
    return {}


def save_processed_log(watch_dir: Path, log: dict) -> None:
    log_path = watch_dir / PROCESSED_LOG
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def file_hash(path: Path) -> str:
    """Content hash so the same PDF re-added under a new name is still
    recognised as already-ingested, and an edited-then-resaved file with
    the same name is correctly treated as new."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def wait_until_stable(path: Path) -> bool:
    """Poll file size until it stops changing, to avoid uploading a
    half-written file. Returns False if the file disappeared."""
    last_size = -1
    stable_rounds = 0
    while stable_rounds < STABLE_CHECK_ROUNDS:
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == last_size and size > 0:
            stable_rounds += 1
        else:
            stable_rounds = 0
        last_size = size
        time.sleep(STABLE_CHECK_INTERVAL)
    return True


def ingest_file(path: Path, api_url: str, endpoint: str) -> bool:
    """POST the PDF to FastAPI's ingestion endpoint. Retries with backoff."""
    url = api_url.rstrip("/") + endpoint

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "application/pdf")}
                response = requests.post(url, files=files, timeout=60)

            if response.status_code in (200, 201):
                logger.info("Ingested %s (status %s)", path.name, response.status_code)
                return True

            logger.warning(
                "Ingestion failed for %s: HTTP %s - %s (attempt %d/%d)",
                path.name, response.status_code, response.text[:200], attempt, MAX_RETRIES,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(
                "Request error ingesting %s: %s (attempt %d/%d)",
                path.name, e, attempt, MAX_RETRIES,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # linear backoff

    logger.error("Giving up on %s after %d attempts", path.name, MAX_RETRIES)
    return False


def handle_new_pdf(path: Path, watch_dir: Path, api_url: str, endpoint: str, log: dict) -> None:
    if not wait_until_stable(path):
        logger.warning("%s disappeared before it became stable, skipping", path.name)
        return

    digest = file_hash(path)
    if digest in log:
        logger.info("Skipping %s - already ingested (matches %s)", path.name, log[digest]["original_name"])
        return

    logger.info("New PDF detected: %s", path.name)
    success = ingest_file(path, api_url, endpoint)

    dest_dir = watch_dir / (PROCESSED_DIR_NAME if success else FAILED_DIR_NAME)
    dest_dir.mkdir(exist_ok=True)
    dest_path = dest_dir / path.name
    try:
        shutil.move(str(path), str(dest_path))
    except OSError as e:
        logger.warning("Could not move %s to %s: %s", path.name, dest_dir, e)

    if success:
        log[digest] = {
            "original_name": path.name,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_processed_log(watch_dir, log)


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------

class PDFHandler(FileSystemEventHandler):
    def __init__(self, watch_dir: Path, api_url: str, endpoint: str, log: dict):
        self.watch_dir = watch_dir
        self.api_url = api_url
        self.endpoint = endpoint
        self.log = log

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return
        # Skip files already inside the processed/failed subfolders
        if PROCESSED_DIR_NAME in path.parts or FAILED_DIR_NAME in path.parts:
            return
        handle_new_pdf(path, self.watch_dir, self.api_url, self.endpoint, self.log)


# ---------------------------------------------------------------------------
# Startup sweep — catch any PDFs already sitting in the folder before
# the watcher started (e.g. after a restart)
# ---------------------------------------------------------------------------

def sweep_existing(watch_dir: Path, api_url: str, endpoint: str, log: dict) -> None:
    existing = [
        p for p in watch_dir.glob("*.pdf")
        if p.is_file()
    ]
    if not existing:
        return
    logger.info("Startup sweep: found %d existing PDF(s) to check", len(existing))
    for path in existing:
        handle_new_pdf(path, watch_dir, api_url, endpoint, log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Watch a folder and auto-ingest new PDFs into the RAG pipeline.")
    parser.add_argument("--watch-dir", default="./incoming_pdfs", help="Folder to monitor for new PDFs")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the FastAPI backend")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Ingestion endpoint path")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the startup sweep of existing files")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                         help="Seconds between folder scans (polling observer, works reliably in containers/network drives)")
    args = parser.parse_args()

    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    log = load_processed_log(watch_dir)

    if not args.no_sweep:
        sweep_existing(watch_dir, args.api_url, args.endpoint, log)

    handler = PDFHandler(watch_dir, args.api_url, args.endpoint, log)
    observer = Observer(timeout=args.poll_interval)
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    logger.info("Watching %s for new PDFs -> %s%s", watch_dir.resolve(), args.api_url, args.endpoint)
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped.")
    observer.join()


if __name__ == "__main__":
    main()
