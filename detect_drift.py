"""
Stage 2 - Drift Detection
Compares the latest batch in replay_log.csv against baseline.csv,
flags similarity drops and hash mismatches, and classifies each
flagged question as content drift, embedding drift, or query drift.
Saves the results to drift_report.csv.
"""

import pandas as pd

BASELINE_CSV = "baseline.csv"
REPLAY_LOG_CSV = "replay_log.csv"
DRIFT_REPORT_CSV = "drift_report.csv"

SIMILARITY_DROP_THRESHOLD = 0.15   # flag if similarity changes by more than this
QUERY_GAP_THRESHOLD = 0.70         # below this similarity = "no strong match"
EMBEDDING_DRIFT_FRACTION = 0.5
MEAN_SHIFT_THRESHOLD = 0.05        # if the average similarity across ALL questions shifts by more than this, flag embedding drift     # if >=50% of questions shift together, it's embedding drift


def load_latest_batch():
    replay = pd.read_csv(REPLAY_LOG_CSV)
    latest_timestamp = replay["timestamp"].max()
    return replay[replay["timestamp"] == latest_timestamp].reset_index(drop=True), latest_timestamp


def main():
    baseline = pd.read_csv(BASELINE_CSV)
    latest, latest_timestamp = load_latest_batch()

    merged = baseline.merge(
        latest,
        on="question",
        suffixes=("_baseline", "_latest")
    )

    merged["similarity_drop"] = merged["similarity_baseline"] - merged["similarity_latest"]
    merged["abs_similarity_change"] = merged["similarity_drop"].abs()
    merged["hash_changed"] = merged["doc_hash_baseline"] != merged["doc_hash_latest"]
    merged["doc_changed"] = merged["matched_doc_baseline"] != merged["matched_doc_latest"]
    merged["flagged"] = (merged["abs_similarity_change"] > SIMILARITY_DROP_THRESHOLD) | merged["hash_changed"] | merged["doc_changed"]

    total_questions = len(merged)
    questions_with_big_change = (merged["abs_similarity_change"] > SIMILARITY_DROP_THRESHOLD).sum()
    # Only count a hash change as "real content drift noise" if the matched
    # document stayed the same. If the matched doc itself changed, that's a
    # retrieval-target shift, not a content edit, so it shouldn't block the
    # embedding-drift check.
    any_hash_changed = (merged["hash_changed"] & ~merged["doc_changed"]).any()
    mean_abs_shift = merged["abs_similarity_change"].mean() if total_questions > 0 else 0

    looks_like_embedding_drift = (
        not any_hash_changed
        and total_questions > 0
        and (
            (questions_with_big_change / total_questions) >= EMBEDDING_DRIFT_FRACTION
            or mean_abs_shift >= MEAN_SHIFT_THRESHOLD
        )
    )

    def classify(row):
        # Retrieval target changed entirely - the pipeline now fetches a
        # DIFFERENT document for this question. This is a serious embedding
        # drift symptom, not content drift (the original doc's content is fine).
        if row["doc_changed"]:
            return "EMBEDDING_DRIFT"

        # Content drift: same document matched both times, but ITS hash changed
        if row["hash_changed"]:
            return "CONTENT_DRIFT"

        # Embedding drift: most questions shifted together, no genuine hash changes anywhere
        if looks_like_embedding_drift and row["abs_similarity_change"] > 0.03:
            return "EMBEDDING_DRIFT"

        # Query drift: consistently weak match, in both baseline and now (a coverage gap)
        if row["similarity_baseline"] < QUERY_GAP_THRESHOLD and row["similarity_latest"] < QUERY_GAP_THRESHOLD:
            return "QUERY_DRIFT"

        # Flagged for a similarity change, but doesn't fit a clean pattern above
        if row["flagged"]:
            return "UNKNOWN_DRIFT"

        return "NO_DRIFT"

    merged["drift_type"] = merged.apply(classify, axis=1)

    report_columns = [
        "question",
        "matched_doc_baseline",
        "matched_doc_latest",
        "similarity_baseline",
        "similarity_latest",
        "similarity_drop",
        "hash_changed",
        "doc_changed",
        "drift_type"
    ]
    report = merged[report_columns].rename(columns={"matched_doc_baseline": "matched_doc"})
    report.to_csv(DRIFT_REPORT_CSV, index=False)

    print(f"Drift report for replay batch at {latest_timestamp}\n")
    print(report.to_string(index=False))

    drifted = report[report["drift_type"] != "NO_DRIFT"]
    print(f"\n{len(drifted)} of {len(report)} questions show drift.")
    if not drifted.empty:
        print("\nBreakdown by drift type:")
        print(drifted["drift_type"].value_counts().to_string())

    print(f"\nSaved full report to {DRIFT_REPORT_CSV}")


if __name__ == "__main__":
    main()