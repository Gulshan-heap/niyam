"""Evaluation dashboard: retrieval quality across saved runs (python -m niyam.evaluation)."""

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("NIYAM_API_URL", "http://localhost:8000")
# Categorical slots 1-3 of the reference palette (validated as a set, light surface).
METRICS = {"MRR": "mrr", "hit@1": "hit@1", "hit@10": "hit@10"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

st.set_page_config(page_title="Niyam · Evaluation", page_icon="📊", layout="wide")
st.title("Retrieval evaluation")
st.caption(
    "Hand-written questions with the RBI documents that answer them (eval/golden.jsonl). "
    "Each run is stored with its git commit; 'dirty' means uncommitted code."
)

try:
    runs = httpx.get(f"{API_URL}/eval/runs", timeout=30).json()
except httpx.HTTPError as exc:
    st.error(f"Could not load runs from the API: {exc}")
    st.stop()
if not runs:
    st.info("No runs yet. Run `uv run python -m niyam.evaluation`.")
    st.stop()

rows = []
for r in runs:
    s = r["summary"]
    rows.append(
        {
            "run": f"#{r['id']}",
            "retriever": r["retriever"],
            "date filter": "yes" if r["apply_as_of"] else "no",
            "commit": (r["git_sha"] or "")[:7] + (" (dirty)" if r["git_dirty"] else ""),
            "questions": r["questions"],
            **{label: s.get(key) for label, key in METRICS.items()},
            "in force@5": s.get("in_force@5"),
            "when": r["created_at"][:16].replace("T", " "),
        }
    )
df = pd.DataFrame(rows)

# Headline: the latest run with Niyam's real setting (date filter on), not an ablation.
filtered = df[df["date filter"] == "yes"]
latest = (filtered if len(filtered) else df).iloc[-1]
st.caption(f"Headline numbers: run {latest['run']} ({latest['retriever']}, date filter on)")
c1, c2, c3 = st.columns(3)
c1.metric("MRR", f"{latest['MRR']:.3f}")
c2.metric("hit@10", f"{latest['hit@10']:.3f}")
in_force = latest["in force@5"]
c3.metric("In force@5", "–" if pd.isna(in_force) else f"{in_force:.0%}")

st.subheader("Across runs")
chart = df.set_index("run")[list(METRICS)]
st.line_chart(chart, color=COLORS, y_label="score (0–1)", x_label="run")

st.subheader("All runs")
st.dataframe(df, hide_index=True, width="stretch")
