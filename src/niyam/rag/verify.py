"""Check that every cited quote really appears in the passage it cites.

The model is asked to support each sentence with verbatim quotes from numbered passages.
Models paraphrase and sometimes invent, so each quote is fuzzy-matched against its
passage; a match gives the exact span, which becomes absolute offsets into the document
(chunk.char_start + local offset) for highlighting. Sentences left with no verified quote
are marked unsupported, and an answer with no supported sentence is an abstention.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

MATCH_THRESHOLD = 90  # rapidfuzz partial_ratio, 0-100
MIN_QUOTE_CHARS = 12  # shorter quotes match by chance


@dataclass
class PassageRef:
    """A retrieved passage as shown to the model ([P1], [P2], ...)."""

    label: str
    chunk_id: int
    source_id: str | None
    title: str
    heading: str | None
    text: str
    char_start: int  # offset of this passage in the document's raw_text
    url: str
    issued_date: str | None = None
    in_force: bool | None = None


@dataclass
class Citation:
    label: str
    quote: str
    verified: bool
    score: float
    chunk_id: int | None = None
    source_id: str | None = None
    doc_char_start: int | None = None  # absolute span in the document, when verified
    doc_char_end: int | None = None


@dataclass
class Sentence:
    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return any(c.verified for c in self.citations)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"').replace("’", "'")).strip()


def locate(quote: str, passage: str) -> tuple[float, int | None, int | None]:
    """Best fuzzy match of `quote` inside `passage`: (score, start, end) in passage offsets."""
    q = _norm(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return 0.0, None, None
    exact = passage.find(quote)
    if exact >= 0:
        return 100.0, exact, exact + len(quote)
    align = fuzz.partial_ratio_alignment(q, passage, score_cutoff=0)
    if align is None:
        return 0.0, None, None
    return align.score, align.dest_start, align.dest_end


def verify_sentences(raw: list[dict], passages: list[PassageRef]) -> list[Sentence]:
    """Turn the model's [{"text", "citations": [{"passage", "quote"}]}] into checked Sentences."""
    by_label = {p.label: p for p in passages}
    sentences = []
    for item in raw:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cites = []
        for c in item.get("citations") or []:
            label = str(c.get("passage", "")).strip().strip("[]")
            quote = str(c.get("quote", "")).strip()
            p = by_label.get(label)
            if p is None:
                cites.append(Citation(label, quote, verified=False, score=0.0))
                continue
            score, start, end = locate(quote, p.text)
            ok = score >= MATCH_THRESHOLD and start is not None
            cites.append(
                Citation(
                    label=label,
                    quote=quote,
                    verified=ok,
                    score=score,
                    chunk_id=p.chunk_id,
                    source_id=p.source_id,
                    doc_char_start=p.char_start + start if ok else None,
                    doc_char_end=p.char_start + end if ok else None,
                )
            )
        sentences.append(Sentence(text, cites))
    return sentences
