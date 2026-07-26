"""
Stage 0 - Build Baseline
Embeds the demo docs into ChromaDB, runs 15 fixed questions against them,
and saves the results to baseline.csv for future drift comparison.
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
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # used for baseline; swapped later to simulate embedding drift
BASELINE_CSV = "baseline.csv"

# 15 fixed questions - covers all 5 docs, plus 1 "dental benefits" question
# that has NO matching doc, to simulate query drift later.
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
    "Does the company cover dental benefits?"  # <- deliberate gap, no doc covers this
]


def compute_file_hash(filepath):
    """Returns SHA256 hash of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_documents():
    """Reads all .txt files from the docs folder and returns their content + hash."""
    docs = {}
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs[filename] = {
                "content": content,
                "hash": compute_file_hash(filepath)
            }
    return docs


def main():
    print("Loading documents from docs/ ...")
    docs = load_documents()
    print(f"Found {len(docs)} documents: {list(docs.keys())}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("Setting up ChromaDB collection ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection if it exists, so re-running this script is safe/repeatable
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    print("Embedding documents into ChromaDB ...")
    collection.add(
        ids=list(docs.keys()),
        documents=[d["content"] for d in docs.values()],
        metadatas=[{"filename": name, "hash": d["hash"]} for name, d in docs.items()]
    )

    print(f"Running {len(QUESTIONS)} baseline questions ...")
    results = []
    timestamp = datetime.datetime.now().isoformat()

    for question in QUESTIONS:
        query_result = collection.query(
            query_texts=[question],
            n_results=1
        )

        matched_doc = query_result["ids"][0][0]
        distance = query_result["distances"][0][0]
        # ChromaDB default distance is squared L2 for this embedding function's normalized vectors,
        # which behaves like cosine distance here. We convert to a similarity score (higher = better match).
        similarity = 1 - (distance / 2)
        doc_hash = query_result["metadatas"][0][0]["hash"]

        results.append({
            "timestamp": timestamp,
            "question": question,
            "matched_doc": matched_doc,
            "similarity": round(similarity, 4),
            "doc_hash": doc_hash,
            "embedding_model": EMBEDDING_MODEL_NAME
        })

    df = pd.DataFrame(results)
    df.to_csv(BASELINE_CSV, index=False)

    print(f"\nDone. Saved {len(df)} rows to {BASELINE_CSV}")
    print(df[["question", "matched_doc", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()