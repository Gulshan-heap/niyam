"""Niyam UI: ask a question about RBI rules, optionally as of a past date, and see each
sentence of the answer with the exact source passage highlighted.

    uv run --group ui streamlit run ui/app.py
NIYAM_API_URL points at the API (default http://localhost:8000).
"""

import html
import os
from datetime import date

import httpx
import streamlit as st

API_URL = os.environ.get("NIYAM_API_URL", "http://localhost:8000")
CONTEXT_CHARS = 700  # document text shown around a highlighted quote

st.set_page_config(page_title="Niyam", page_icon="⚖️", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_document(source_id: str) -> dict:
    r = httpx.get(f"{API_URL}/documents/{source_id}", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_timeline(source_id: str) -> dict:
    r = httpx.get(f"{API_URL}/documents/{source_id}/timeline", timeout=30)
    r.raise_for_status()
    return r.json()


def ask(question: str, as_of: date | None) -> dict:
    payload = {"question": question, "as_of": as_of.isoformat() if as_of else None}
    r = httpx.post(f"{API_URL}/ask", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def highlighted_excerpt(text: str, start: int, end: int) -> str:
    """HTML: the quote marked, with some surrounding text for context."""
    a, b = max(0, start - CONTEXT_CHARS), min(len(text), end + CONTEXT_CHARS)
    before, quote, after = text[a:start], text[start:end], text[end:b]
    body = (
        ("…" if a > 0 else "")
        + html.escape(before)
        + f"<mark>{html.escape(quote)}</mark>"
        + html.escape(after)
        + ("…" if b < len(text) else "")
    )
    return f'<div style="white-space: pre-wrap; font-size: 0.9rem">{body}</div>'


st.title("Niyam")
st.caption("Which RBI rule is in force on a given date? Answers quote their sources.")

with st.form("ask"):
    question = st.text_input(
        "Question", placeholder="What is the minimum cooling-off period for digital loans?"
    )
    col1, col2 = st.columns([1, 3])
    use_date = col1.checkbox("As of a past date")
    as_of = col2.date_input("Date", value=date.today(), disabled=not use_date)
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching circulars and drafting an answer..."):
        try:
            result = ask(question.strip(), as_of if use_date else None)
        except httpx.HTTPError as exc:
            st.error(f"The API call failed: {exc}")
            st.stop()

    source = {
        "question": "read from your question",
        "request": "the date you chose",
        "today": "today",
    }.get(result.get("as_of_source", ""), "")
    st.caption(f"Rules in force on {result['as_of']} ({source})")
    with st.expander("How this answer was found"):
        for step in result.get("trace", []):
            st.text(step)

    if result["abstained"]:
        st.warning(result.get("note") or "The sources found don't answer this question.")
    else:
        labels = {p["label"]: p for p in result["passages"]}
        for s in result["sentences"]:
            refs = sorted({c["passage"] for c in s["citations"] if c["verified"]})
            st.markdown(f"{s['text']} " + " ".join(f"`[{r}]`" for r in refs))
        if result["unsupported"]:
            with st.expander(f"{len(result['unsupported'])} sentence(s) dropped: no source quote"):
                for t in result["unsupported"]:
                    st.write(t)

        st.subheader("Sources")
        for s in result["sentences"]:
            for c in s["citations"]:
                if not c["verified"] or c["source_id"] is None:
                    continue
                p = labels.get(c["passage"], {})
                status = "in force" if p.get("in_force") else "not in force on that date"
                with st.expander(f"[{c['passage']}] {p.get('title', c['source_id'])} ({status})"):
                    if p.get("heading"):
                        st.caption(p["heading"])
                    doc = fetch_document(c["source_id"])
                    st.markdown(
                        highlighted_excerpt(
                            doc["raw_text"] or "", c["doc_char_start"], c["doc_char_end"]
                        ),
                        unsafe_allow_html=True,
                    )
                    meta = f"Issued {doc['issued_date']}"
                    if doc.get("text_as_of") and doc["text_as_of"] != doc["issued_date"]:
                        meta += f" · text as of {doc['text_as_of']}"
                    if doc.get("is_withdrawn"):
                        meta += " · withdrawn" + (
                            f" on {doc['withdrawn_on']}" if doc.get("withdrawn_on") else ""
                        )
                    st.caption(f"{meta} · [open on rbi.org.in]({doc['url']})")
                    timeline = fetch_timeline(c["source_id"])
                    if timeline["amended_by"]:
                        lines = [
                            f"- {a['issued_date']}: [{a['title']}]({a['url']})"
                            + (" *(withdrawn)*" if a["is_withdrawn"] else "")
                            for a in timeline["amended_by"]
                        ]
                        st.markdown("**Amended by**\n" + "\n".join(lines))
