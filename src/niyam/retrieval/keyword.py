"""Document-level keyword search over `documents.tsv` (title weighted above body).

Two query modes:
- "any" (default): a document matches if it contains any query term; ranking rewards
  documents that contain more of them, close together. Suits natural-language questions.
- "all": web-search syntax (every term required, "quoted phrases", -exclusions).
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from sqlalchemy import Text, and_, cast, func, literal, or_, select
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.orm import Session

from niyam.db.models import Document

Mode = Literal["any", "all"]

# ts_rank_cd normalization flags: 1 divides by 1 + log(document length), so long Master
# Directions don't win just by repeating common words; 4 divides by the mean harmonic
# distance between matched extents, rewarding query terms that appear close together.
# 1|4 was best on the golden set (MRR 0.65 vs 0.41 for 1 alone; see eval_runs).
RANK_NORMALIZATION = 5
HEADLINE_OPTIONS = "MaxFragments=2, MaxWords=30, MinWords=12, StartSel=«, StopSel=»"


@dataclass
class SearchFilters:
    as_of: date | None = None  # only documents in force on this date
    doc_type: str | None = None
    department: str | None = None  # case-insensitive substring
    issued_from: date | None = None
    issued_to: date | None = None


@dataclass
class SearchHit:
    doc_id: int
    source_id: str | None
    title: str
    circular_no: str | None
    doc_type: str
    department: str | None
    issued_date: date
    is_withdrawn: bool
    withdrawn_on: date | None
    url: str
    score: float
    snippet: str


def build_tsquery(query: str, mode: Mode = "any"):
    if mode == "all":
        return func.websearch_to_tsquery("english", query)
    # plainto_tsquery normalises/stems the terms and joins them with &; swap to |.
    anded = cast(func.plainto_tsquery("english", query), Text)
    return cast(func.replace(anded, "&", "|"), TSQUERY)


def _filter_clauses(f: SearchFilters) -> list:
    clauses = []
    if f.as_of is not None:
        clauses.append(Document.valid_from <= f.as_of)
        clauses.append(or_(Document.valid_to.is_(None), Document.valid_to > f.as_of))
        if f.as_of >= date.today():
            # Marked withdrawn but with no known date: certainly not in force any more.
            clauses.append(Document.is_withdrawn.is_(False))
    if f.doc_type:
        clauses.append(Document.doc_type == f.doc_type)
    if f.department:
        clauses.append(Document.department.ilike(f"%{f.department}%"))
    if f.issued_from:
        clauses.append(Document.issued_date >= f.issued_from)
    if f.issued_to:
        clauses.append(Document.issued_date <= f.issued_to)
    return clauses


def keyword_search(
    session: Session,
    query: str,
    k: int = 10,
    filters: SearchFilters | None = None,
    mode: Mode = "any",
    normalization: int = RANK_NORMALIZATION,
) -> list[SearchHit]:
    tsq = build_tsquery(query, mode)
    if session.scalar(select(func.numnode(tsq))) == 0:
        return []  # only stopwords / punctuation

    score = func.ts_rank_cd(Document.tsv, tsq, literal(normalization)).label("score")
    ranked = (
        select(Document.id, score)
        .where(Document.tsv.op("@@")(tsq), and_(*_filter_clauses(filters or SearchFilters())))
        .order_by(score.desc(), Document.issued_date.desc())
        .limit(k)
        .subquery()
    )
    # Headlines are costly on long documents, so only build them for the top k.
    snippet = func.ts_headline("english", Document.raw_text, tsq, HEADLINE_OPTIONS)
    rows = session.execute(
        select(Document, ranked.c.score, snippet)
        .join(ranked, ranked.c.id == Document.id)
        .order_by(ranked.c.score.desc(), Document.issued_date.desc())
    ).all()
    return [
        SearchHit(
            doc_id=d.id,
            source_id=d.source_id,
            title=d.title,
            circular_no=d.circular_no,
            doc_type=d.doc_type,
            department=d.department,
            issued_date=d.issued_date,
            is_withdrawn=d.is_withdrawn,
            withdrawn_on=d.withdrawn_on,
            url=d.url,
            score=float(s),
            snippet=" ".join(snip.split()),
        )
        for d, s, snip in rows
    ]
