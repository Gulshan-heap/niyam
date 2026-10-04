"""Prompt for grounded answers with per-sentence quotes."""

import json

from niyam.rag.verify import PassageRef

SYSTEM = """You answer questions about Indian financial regulation (Reserve Bank of India \
circulars and directions) using ONLY the numbered passages provided.

Rules:
- Every sentence of your answer must be supported by at least one passage. For each \
sentence give the passage label and a short VERBATIM quote (copied exactly, 10-40 words) \
from that passage that supports it.
- Do not use outside knowledge. If the passages do not answer the question, set \
"abstained": true and explain briefly in one sentence without citations.
- Prefer passages from rules in force on the date asked about; mention dates and \
circular names when they matter.
- Be concise: at most 6 sentences.

Reply with JSON only, in this shape:
{"abstained": false, "sentences": [{"text": "...", "citations": [{"passage": "P1", \
"quote": "..."}]}]}"""


def format_passages(passages: list[PassageRef]) -> str:
    blocks = []
    for p in passages:
        meta = [p.title]
        if p.heading:
            meta.append(p.heading)
        if p.issued_date:
            meta.append(f"issued {p.issued_date}")
        if p.in_force is not None:
            meta.append("in force" if p.in_force else "not in force on the date asked")
        blocks.append(f"[{p.label}] {' | '.join(meta)}\n{p.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, passages: list[PassageRef], as_of: str | None) -> list[dict]:
    when = f"The question is about the rules in force on {as_of}." if as_of else ""
    user = f"Passages:\n\n{format_passages(passages)}\n\nQuestion: {question}\n{when}".strip()
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_reply(text: str) -> dict:
    """The model's JSON, tolerating code fences or prose around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model reply is not a JSON object")
    data.setdefault("sentences", [])
    data.setdefault("abstained", False)
    return data
