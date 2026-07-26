"""
Simulates embedding drift by re-embedding ALL documents in ChromaDB
using a DIFFERENT embedding model than the one used for baseline.csv.

This mimics a real-world scenario: someone upgrades or changes the
embedding model in production, causing the vector space to shift for
every document at once - even though no document content changed.
"""

import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions

DOCS_FOLDER = "docs"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "hr_it_docs"

# The NEW model we're swapping to - deliberately different from
# all-MiniLM-L6-v2 (used in build_baseline.py) to simulate drift.
NEW_EMBEDDING_MODEL_NAME = "average_word_embeddings_glove.6B.300d"

def compute_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    print(f"Loading NEW embedding model: {NEW_EMBEDDING_MODEL_NAME} ...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=NEW_EMBEDDING_MODEL_NAME
    )

    print("Connecting to ChromaDB and deleting old collection ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    print("Re-embedding all documents with the new model ...")
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

    collection.add(
        ids=list(docs.keys()),
        documents=[d["content"] for d in docs.values()],
        metadatas=[{"filename": name, "hash": d["hash"]} for name, d in docs.items()]
    )

    print(f"\nDone. ChromaDB now uses '{NEW_EMBEDDING_MODEL_NAME}' for all {len(docs)} documents.")
    print("Note: document CONTENT and hashes are unchanged - only the embedding model changed.")
    print(f"\nIMPORTANT: replay.py's EMBEDDING_MODEL_NAME must also be updated to")
    print(f"'{NEW_EMBEDDING_MODEL_NAME}' so replay uses the same model as what's now in ChromaDB.")


if __name__ == "__main__":
    main()