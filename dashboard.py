"""
Stage 5 - DriftSentinel Dashboard
A Streamlit app that visualizes:
- Drift-over-time chart (similarity trends per question/document)
- Current status table (latest drift_report.csv)
- Diagnosis + fix feed (remediation_log.csv)
- Diff view (compare baseline vs latest doc content, or two replay batches)
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASELINE_CSV = "baseline.csv"
REPLAY_LOG_CSV = "replay_log.csv"
DRIFT_REPORT_CSV = "drift_report.csv"
REMEDIATION_LOG_CSV = "remediation_log.csv"
CONTENT_GAP_REPORT_CSV = "content_gap_report.csv"
DOCS_FOLDER = "docs"
DRIFT_COLORS = {
    "CONTENT_DRIFT": "#fff3cd",
    "EMBEDDING_DRIFT": "#f8d7da",
    "QUERY_DRIFT": "#cfe2ff",
    "UNKNOWN_DRIFT": "#e2e3e5",
    "NO_DRIFT": "#d1e7dd",
}
DOC_COLORS = {
    "leave_policy.txt": "#4C78A8",
    "password_reset.txt": "#F58518",
    "expense_reimbursement.txt": "#54A24B",
    "remote_work_policy.txt": "#E45756",
    "benefits_enrollment.txt": "#B279A2",
}

st.set_page_config(page_title="DriftSentinel", layout="wide")


def load_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    st.title("🛰️ DriftSentinel")
    st.caption("Monitoring RAG retrieval drift over time — HR/IT Knowledge Base demo")

    baseline = load_csv_safe(BASELINE_CSV)
    replay_log = load_csv_safe(REPLAY_LOG_CSV)
    drift_report = load_csv_safe(DRIFT_REPORT_CSV)
    remediation_log = load_csv_safe(REMEDIATION_LOG_CSV)
    content_gaps = load_csv_safe(CONTENT_GAP_REPORT_CSV)

    if baseline.empty or replay_log.empty:
        st.warning("No data yet. Run build_baseline.py and replay.py first.")
        return

    # ---- Top-level summary metrics ----
    latest_timestamp = replay_log["timestamp"].max()
    total_runs = replay_log["timestamp"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Questions Monitored", baseline["question"].nunique())
    col2.metric("Replay Runs Logged", total_runs)

    if not drift_report.empty:
        drifted_count = (drift_report["drift_type"] != "NO_DRIFT").sum()
        col3.metric("Questions Currently Drifted", drifted_count, delta=None)
    else:
        col3.metric("Questions Currently Drifted", "—")

    col4.metric("Last Replay Run", latest_timestamp)

    st.divider()

    # ---- TABS ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Drift Over Time",
        "📋 Current Status",
        "🛠️ Diagnosis & Fixes",
        "🔍 Diff Viewer"
    ])

    # ================= TAB 1: Drift Over Time =================
    with tab1:
        st.subheader("Similarity score over time, per question")

        questions = sorted(replay_log["question"].unique())
        selected_questions = st.multiselect(
            "Select questions to plot (default: all)",
            questions,
            default=questions
        )

        history = replay_log[replay_log["question"].isin(selected_questions)].copy()

        # Prepend baseline as "time zero" for each question
        baseline_points = baseline[baseline["question"].isin(selected_questions)].copy()
        baseline_points["timestamp"] = "BASELINE"

        combined = pd.concat([baseline_points, history], ignore_index=True)
        combined = combined.sort_values("timestamp")

        fig = px.line(
            combined,
            x="timestamp",
            y="similarity",
            color="question",
            markers=True,
            title="Similarity score across baseline + replay runs"
        )
        fig.add_hline(
            y=0.70, line_dash="dot", line_color="orange",
            annotation_text="Query drift threshold (0.70)"
        )
        fig.update_layout(xaxis_title="Run", yaxis_title="Similarity", legend_title="Question")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matched document over time")
        doc_history = combined.groupby(["question", "matched_doc"]).size().reset_index(name="count")
        fig2 = px.bar(
            doc_history, x="question", y="count", color="matched_doc",
            color_discrete_map=DOC_COLORS,
            title="Which document each question matched, across all runs"
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    # ================= TAB 2: Current Status =================
    with tab2:
        st.subheader("Latest Drift Report")

        if drift_report.empty:
            st.info("No drift report yet. Run detect_drift.py first.")
        else:
            def highlight_drift(row):
                hex_color = DRIFT_COLORS.get(row["drift_type"], "")
                style = f"background-color: {hex_color}" if hex_color else ""
                return [style] * len(row)

            st.dataframe(
                drift_report.style.apply(highlight_drift, axis=1),
                use_container_width=True,
                height=500
            )

            st.subheader("Drift type breakdown")
            counts = drift_report["drift_type"].value_counts().reset_index()
            counts.columns = ["drift_type", "count"]
            fig3 = px.pie(
                counts,
                names="drift_type",
                values="count",
                color="drift_type",
                color_discrete_map=DRIFT_COLORS,
                title="Current drift breakdown",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ================= TAB 3: Diagnosis & Fixes =================
    with tab3:
        st.subheader("Remediation Feed")

        if remediation_log.empty:
            st.info("No remediation actions logged yet. Run remediate.py first.")
        else:
            for _, row in remediation_log.sort_values("timestamp", ascending=False).iterrows():
                icon = {
                    "AUTO_RESOLVED": "✅",
                    "PENDING_HUMAN_APPROVAL": "⏳",
                    "PENDING_HUMAN_INPUT": "✍️",
                    "NEEDS_MANUAL_REVIEW": "⚠️"
                }.get(row["status"], "•")

                with st.container(border=True):
                    st.markdown(f"{icon} **{row['question']}**")
                    st.caption(f"Document: `{row['matched_doc']}` · Drift type: `{row['drift_type']}` · Status: `{row['status']}`")
                    st.write(row["action_taken"])

        if not content_gaps.empty:
            st.subheader("📝 Content Gap Report (needs human-authored content)")
            st.dataframe(content_gaps, use_container_width=True)

    # ================= TAB 4: Diff Viewer =================
    with tab4:
        st.subheader("Compare document versions")
        st.caption("Shows the CURRENT on-disk document content. Useful for reviewing what changed after a content-drift alert.")

        doc_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")]
        selected_doc = st.selectbox("Select a document", sorted(doc_files))

        if selected_doc:
            filepath = os.path.join(DOCS_FOLDER, selected_doc)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            st.text_area("Current content on disk", content, height=300)

            # Show baseline hash vs current hash for this doc, if available
            if not baseline.empty:
                doc_rows = baseline[baseline["matched_doc"] == selected_doc]
                if not doc_rows.empty:
                    st.caption(f"Baseline hash on record: `{doc_rows.iloc[0]['doc_hash']}`")

            import hashlib
            with open(filepath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            st.caption(f"Current on-disk hash: `{current_hash}`")


if __name__ == "__main__":
    main()