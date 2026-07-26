"""
Stage 3 - Remediation
Reads drift_report.csv and takes action based on drift_type:
- CONTENT_DRIFT  -> auto-fix: re-embed the affected document(s) in ChromaDB
- EMBEDDING_DRIFT -> suggest rollback to previous embedding model (human approves)
- QUERY_DRIFT    -> suggest a content-gap report entry (human fills in)
- UNKNOWN_DRIFT  -> flagged for manual review, no automated action taken

Logs every action taken (or suggested) to remediation_log.csv.
"""

import os
import hashlib
import datetime
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

DOCS_FOLDER = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "hr_it_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DRIFT_REPORT_CSV = "drift_report.csv"
REMEDIATION_LOG_CSV = "remediation_log.csv"
CONTENT_GAP_REPORT_CSV = "content_gap_report.csv"
BASELINE_CSV = "baseline.csv"


def compute_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def reembed_document(collection, filename):
    """Re-reads a single doc from disk and updates its embedding + hash in ChromaDB."""
    filepath = os.path.join(DOCS_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    new_hash = compute_file_hash(filepath)

    collection.update(
        ids=[filename],
        documents=[content],
        metadatas=[{"filename": filename, "hash": new_hash}]
    )
    return new_hash


def main():
    report = pd.read_csv(DRIFT_REPORT_CSV)
    drifted = report[report["drift_type"] != "NO_DRIFT"]

    if drifted.empty:
        print("No drift detected. Nothing to remediate.")
        return

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    timestamp = datetime.datetime.now().isoformat()
    log_entries = []
    gap_entries = []
    reembedded_docs = set()

    for _, row in drifted.iterrows():
        drift_type = row["drift_type"]
        doc = row["matched_doc"]
        question = row["question"]

        if drift_type == "CONTENT_DRIFT":
            if doc not in reembedded_docs:
                new_hash = reembed_document(collection, doc)
                reembedded_docs.add(doc)
                action = f"AUTO-FIXED: re-embedded {doc} (new hash {new_hash[:12]}...)"
            else:
                action = f"AUTO-FIXED: {doc} already re-embedded this run"
            log_entries.append({
                "timestamp": timestamp, "question": question, "matched_doc": doc,
                "drift_type": drift_type, "action_taken": action, "status": "AUTO_RESOLVED"
            })

        elif drift_type == "EMBEDDING_DRIFT":
            action = (
                "SUGGESTED: rollback embedding model to previous version. "
                "Human approval required before applying."
            )
            log_entries.append({
                "timestamp": timestamp, "question": question, "matched_doc": doc,
                "drift_type": drift_type, "action_taken": action, "status": "PENDING_HUMAN_APPROVAL"
            })

        elif drift_type == "QUERY_DRIFT":
            action = "SUGGESTED: add a document to cover this topic. Logged to content_gap_report.csv."
            log_entries.append({
                "timestamp": timestamp, "question": question, "matched_doc": doc,
                "drift_type": drift_type, "action_taken": action, "status": "PENDING_HUMAN_INPUT"
            })
            gap_entries.append({
                "timestamp": timestamp,
                "question": question,
                "closest_doc_found": doc,
                "similarity_score": row["similarity_latest"],
                "suggested_fix": "Create or update a document covering this topic"
            })

        else:  # UNKNOWN_DRIFT or anything unexpected
            action = "FLAGGED for manual review. No automated action taken."
            log_entries.append({
                "timestamp": timestamp, "question": question, "matched_doc": doc,
                "drift_type": drift_type, "action_taken": action, "status": "NEEDS_MANUAL_REVIEW"
            })

    if reembedded_docs:
        baseline = pd.read_csv(BASELINE_CSV)
        for doc in reembedded_docs:
            filepath = os.path.join(DOCS_FOLDER, doc)
            new_hash = compute_file_hash(filepath)
            baseline.loc[baseline["matched_doc"] == doc, "doc_hash"] = new_hash
        baseline.to_csv(BASELINE_CSV, index=False)
    log_df = pd.DataFrame(log_entries)
    if os.path.exists(REMEDIATION_LOG_CSV):
        log_df.to_csv(REMEDIATION_LOG_CSV, mode="a", header=False, index=False)
    else:
        log_df.to_csv(REMEDIATION_LOG_CSV, mode="w", header=True, index=False)

    if gap_entries:
        gap_df = pd.DataFrame(gap_entries)
        if os.path.exists(CONTENT_GAP_REPORT_CSV):
            gap_df.to_csv(CONTENT_GAP_REPORT_CSV, mode="a", header=False, index=False)
        else:
            gap_df.to_csv(CONTENT_GAP_REPORT_CSV, mode="w", header=True, index=False)

    print(f"Processed {len(drifted)} drifted question(s):\n")
    print(log_df[["question", "drift_type", "status"]].to_string(index=False))

    if reembedded_docs:
        print(f"\nAuto-fixed (re-embedded) documents: {', '.join(reembedded_docs)}")
    if gap_entries:
        print(f"Logged {len(gap_entries)} content gap(s) to {CONTENT_GAP_REPORT_CSV}")

    print(f"\nFull remediation log saved to {REMEDIATION_LOG_CSV}")


if __name__ == "__main__":
    main()