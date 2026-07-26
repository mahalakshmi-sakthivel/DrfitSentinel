"""
Stage 1 - Scheduled Replay (manual-run version for now)
Re-runs the same fixed questions against the current state of the
ChromaDB collection, and appends results to replay_log.csv.

Each run captures a fresh timestamp, so over multiple runs replay_log.csv
builds up a history we can compare against baseline.csv to detect drift.
"""

import os
import hashlib
import datetime
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ---- CONFIG ----
DOCS_FOLDER = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "hr_it_docs"
EMBEDDING_MODEL_NAME = "average_word_embeddings_glove.6B.300d"  # swapped to simulate embedding drift
REPLAY_LOG_CSV = "replay_log.csv"

# Same 15 questions as build_baseline.py - must stay identical for fair comparison
QUESTIONS = [
    "How many days of paid annual leave do I get?",
    "Can I carry forward unused leave into next year?",
    "How much paid sick leave am I entitled to?",
    "How do I reset my forgotten company password?",
    "What are the password requirements?",
    "What happens if I fail to log in 5 times?",
    "How do I claim reimbursement for a business trip?",
    "Is alcohol reimbursable during business travel?",
    "How long does it take to get my expense claim processed?",
    "How many days per week can I work remotely?",
    "Do I need to inform HR if I work remotely from another city?",
    "Does the company provide equipment for remote work?",
    "When is the open enrollment period for benefits?",
    "Can I add my spouse to my health insurance plan?",
    "Does the company cover dental benefits?"
]


def compute_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_current_doc_hashes():
    """Reads current hash of each doc directly from disk (not from ChromaDB),
    so we can detect if a file changed even before re-embedding."""
    hashes = {}
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            hashes[filename] = compute_file_hash(filepath)
    return hashes


def run_replay():
    print(f"[{datetime.datetime.now().isoformat()}] Starting replay run ...")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Connecting to existing ChromaDB collection ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    current_hashes = get_current_doc_hashes()

    print(f"Running {len(QUESTIONS)} replay questions ...")
    results = []
    timestamp = datetime.datetime.now().isoformat()

    for question in QUESTIONS:
        query_result = collection.query(
            query_texts=[question],
            n_results=1
        )

        matched_doc = query_result["ids"][0][0]
        distance = query_result["distances"][0][0]
        similarity = 1 - (distance / 2)

        # Use the CURRENT on-disk hash for the matched doc, not the hash stored
        # in ChromaDB at embed time - this is what lets us detect content drift
        # even if the doc hasn't been re-embedded yet.
        current_hash = current_hashes.get(matched_doc, "UNKNOWN")

        results.append({
            "timestamp": timestamp,
            "question": question,
            "matched_doc": matched_doc,
            "similarity": round(similarity, 4),
            "doc_hash": current_hash,
            "embedding_model": EMBEDDING_MODEL_NAME
        })

    df_new = pd.DataFrame(results)

    # Append to replay_log.csv if it exists, otherwise create it
    if os.path.exists(REPLAY_LOG_CSV):
        df_new.to_csv(REPLAY_LOG_CSV, mode="a", header=False, index=False)
    else:
        df_new.to_csv(REPLAY_LOG_CSV, mode="w", header=True, index=False)

    print(f"Done. Appended {len(df_new)} rows to {REPLAY_LOG_CSV}")
    print(df_new[["question", "matched_doc", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    from apscheduler.schedulers.blocking import BlockingScheduler

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Manual single run, e.g.: python replay.py --once
        run_replay()
    else:
        # Scheduled mode: runs immediately once, then every 2 hours after that
        print("Starting scheduler. Replay will run now, then every 2 hours.")
        print("Press Ctrl+C to stop.")

        scheduler = BlockingScheduler()
        scheduler.add_job(run_replay, "interval", hours=2, next_run_time=datetime.datetime.now())

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("Scheduler stopped.")
