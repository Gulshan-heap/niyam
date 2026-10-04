"""The hand-written evaluation set: questions and the RBI documents that answer them.

One JSON object per line:

    {"id": "kyc-001", "question": "...", "relevant": ["12943"],
     "as_of": "2024-03-01", "tags": ["kyc", "nbfc"], "note": "why these documents"}

`relevant` holds RBI page ids (documents.source_id). `as_of` is optional: set it when the
question asks about the rule on a past date.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class GoldenQuestion:
    id: str
    question: str
    relevant: list[str]
    as_of: date | None = None
    tags: list[str] = field(default_factory=list)
    note: str | None = None


def load_golden(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            q = GoldenQuestion(
                id=raw["id"],
                question=raw["question"],
                relevant=[str(r) for r in raw["relevant"]],
                as_of=date.fromisoformat(raw["as_of"]) if raw.get("as_of") else None,
                tags=raw.get("tags", []),
                note=raw.get("note"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{lineno}: bad question: {exc}") from exc
        if not q.relevant:
            raise ValueError(f"{path}:{lineno}: {q.id} has no relevant documents")
        if q.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {q.id}")
        seen.add(q.id)
        questions.append(q)
    return questions
