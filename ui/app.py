"""Niyam UI (Flask): ask a question about RBI rules, optionally as of a past date, and see each
sentence of the answer with the exact source passage highlighted. A second page charts the
saved evaluation runs.

    uv run --group ui flask --app ui/app.py run --port 8501
NIYAM_API_URL points at the API (default http://localhost:8000).
"""

import os
import time
from datetime import date

import httpx
from flask import Flask, render_template, request

API_URL = os.environ.get("NIYAM_API_URL", "http://localhost:8000")
CONTEXT_CHARS = 700  # document text shown around a highlighted quote
CACHE_TTL = 3600
SOURCE_LABELS = {
    "question": "read from your question",
    "request": "the date you chose",
    "today": "today",
}
# Categorical slots 1-3 of the reference palette (validated as a set, light surface).
METRICS = {"MRR": "mrr", "hit@1": "hit@1", "hit@10": "hit@10"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

app = Flask(__name__)
_cache: dict[str, tuple[float, dict]] = {}


def _get_json(path: str) -> dict:
    hit = _cache.get(path)
    if hit and time.monotonic() - hit[0] < CACHE_TTL:
        return hit[1]
    r = httpx.get(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    _cache[path] = (time.monotonic(), r.json())
    return _cache[path][1]


def ask(question: str, as_of: str | None) -> dict:
    r = httpx.post(f"{API_URL}/ask", json={"question": question, "as_of": as_of}, timeout=120)
    r.raise_for_status()
    return r.json()


def excerpt(text: str, start: int, end: int) -> dict:
    """The quote with some surrounding text for context (the template marks `quote`)."""
    a, b = max(0, start - CONTEXT_CHARS), min(len(text), end + CONTEXT_CHARS)
    return {
        "before": ("…" if a > 0 else "") + text[a:start],
        "quote": text[start:end],
        "after": text[end:b] + ("…" if b < len(text) else ""),
    }


def build_sources(result: dict) -> list[dict]:
    labels = {p["label"]: p for p in result["passages"]}
    sources = []
    for s in result["sentences"]:
        for c in s["citations"]:
            if not c["verified"] or c["source_id"] is None:
                continue
            p = labels.get(c["passage"], {})
            doc = _get_json(f"/documents/{c['source_id']}")
            meta = f"Issued {doc['issued_date']}"
            if doc.get("text_as_of") and doc["text_as_of"] != doc["issued_date"]:
                meta += f" · text as of {doc['text_as_of']}"
            if doc.get("is_withdrawn"):
                meta += " · withdrawn" + (
                    f" on {doc['withdrawn_on']}" if doc.get("withdrawn_on") else ""
                )
            sources.append(
                {
                    "label": c["passage"],
                    "title": p.get("title", c["source_id"]),
                    "in_force": bool(p.get("in_force")),
                    "heading": p.get("heading"),
                    "excerpt": excerpt(
                        doc["raw_text"] or "", c["doc_char_start"], c["doc_char_end"]
                    ),
                    "meta": meta,
                    "url": doc["url"],
                    "amended_by": _get_json(f"/documents/{c['source_id']}/timeline")["amended_by"],
                }
            )
    return sources


@app.get("/")
def index():
    question = request.args.get("question", "").strip()
    use_date = request.args.get("use_date") == "on"
    as_of = request.args.get("as_of") or date.today().isoformat()
    ctx = {"question": question, "use_date": use_date, "as_of": as_of}
    if question:
        try:
            result = ask(question, as_of if use_date else None)
            ctx.update(
                result=result,
                date_source=SOURCE_LABELS.get(result.get("as_of_source", ""), ""),
                sources=[] if result["abstained"] else build_sources(result),
            )
        except httpx.HTTPError as exc:
            ctx["error"] = f"The API call failed: {exc}"
    return render_template("index.html", **ctx)


def _commit(run: dict) -> str:
    return (run["git_sha"] or "")[:7] + (" (dirty)" if run["git_dirty"] else "")


def _chart(rows: list[dict]) -> dict:
    """Points for an inline SVG line chart: one line per metric, one x step per run."""
    w, h, pad = 640, 240, 36
    step = (w - 2 * pad) / max(len(rows) - 1, 1)
    lines = []
    for (label, _), color in zip(METRICS.items(), COLORS, strict=True):
        pts = [
            (pad + i * step, h - pad - (r[label] or 0) * (h - 2 * pad))
            for i, r in enumerate(rows)
            if r[label] is not None
        ]
        lines.append(
            {"label": label, "color": color, "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}
        )
    return {
        "w": w,
        "h": h,
        "pad": pad,
        "lines": lines,
        "ticks": [{"v": v, "y": h - pad - v * (h - 2 * pad)} for v in (0, 0.25, 0.5, 0.75, 1)],
        "xs": [{"label": r["run"], "x": pad + i * step} for i, r in enumerate(rows)],
    }


@app.get("/evaluation")
def evaluation():
    try:
        runs = httpx.get(f"{API_URL}/eval/runs", timeout=30).json()
    except httpx.HTTPError as exc:
        return render_template("evaluation.html", error=f"Could not load runs from the API: {exc}")
    answer_runs = [r for r in runs if r["retriever"] == "answer-agent"]
    retrieval_runs = [r for r in runs if r["retriever"] != "answer-agent"]
    rows = [
        {
            "run": f"#{r['id']}",
            "retriever": r["retriever"],
            "date_filter": bool(r["apply_as_of"]),
            "commit": _commit(r),
            "questions": r["questions"],
            **{label: r["summary"].get(key) for label, key in METRICS.items()},
            "in_force": r["summary"].get("in_force@5"),
            "when": r["created_at"][:16].replace("T", " "),
        }
        for r in retrieval_runs
    ]
    # Headline: the latest run with Niyam's real setting (date filter on), not an ablation.
    filtered = [r for r in rows if r["date_filter"]]
    headline = (filtered or rows or [None])[-1]
    answers = [
        {
            "run": f"#{r['id']}",
            "commit": _commit(r),
            "questions": r["questions"],
            "answered": r["summary"].get("answered"),
            "grounded": r["summary"].get("grounded"),
            "cites_relevant": r["summary"].get("cites_relevant"),
            "cites_in_force": r["summary"].get("cites_in_force"),
            "when": r["created_at"][:16].replace("T", " "),
        }
        for r in answer_runs
    ]
    return render_template(
        "evaluation.html",
        rows=rows,
        headline=headline,
        chart=_chart(rows) if rows else None,
        answers=answers,
        metric_labels=list(METRICS),
    )


@app.template_filter("pct")
def pct(v):
    return "–" if v is None else f"{v:.0%}"


@app.template_filter("num")
def num(v):
    return "–" if v is None else f"{v:.3f}"
