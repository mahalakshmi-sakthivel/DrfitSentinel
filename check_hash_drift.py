"""
Quick one-off check: compares the doc_hash in baseline.csv against
the doc_hash in the most recent batch of replay_log.csv, per document.
This previews what Stage 2's drift detector will do automatically.
"""

import pandas as pd

baseline = pd.read_csv("baseline.csv")
replay = pd.read_csv("replay_log.csv")

# Get the most recent timestamp's batch from replay_log.csv
latest_timestamp = replay["timestamp"].max()
latest_batch = replay[replay["timestamp"] == latest_timestamp]

# Get one hash per document from baseline (first occurrence is fine, should be same for all rows of that doc)
baseline_hashes = baseline.groupby("matched_doc")["doc_hash"].first()
latest_hashes = latest_batch.groupby("matched_doc")["doc_hash"].first()

print(f"Comparing baseline hashes vs latest replay batch ({latest_timestamp})\n")

for doc in baseline_hashes.index:
    old_hash = baseline_hashes.get(doc, "MISSING")
    new_hash = latest_hashes.get(doc, "MISSING")
    status = "CHANGED" if old_hash != new_hash else "unchanged"
    print(f"{doc:30s}  {status}")
    if status == "CHANGED":
        print(f"   baseline hash: {old_hash}")
        print(f"   latest hash:   {new_hash}")