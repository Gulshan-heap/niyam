"""Chunk-level retrieval: keyword (tsvector), vector (pgvector HNSW), and their fusion.

Hybrid search takes the top candidates from each method and merges them with reciprocal
rank fusion (RRF): score = sum over methods of 1 / (RRF_K + rank). RRF needs no score
calibration between BM25-style ranks and cosine distances, which is why it is the usual
default for hybrid retrieval.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from sqlalchemy import and_, func, literal, or_, select, text
from sqlalchemy.orm import Session

from niyam.db.models import Chunk, Document
from niyam.retrieval.embeddings import Embedder
from niyam.retrieval.keyword import Mode, SearchFilters, build_tsquery

Method = Literal["keyword", "vector", "hybrid"]

RRF_K = 60
# Weight of the keyword list in the fusion (the vector list has 1.0). On the 44-question
# golden set, 0.3 gave the best MRR (0.80 vs 0.78 at 1.0 and 0.78 for vectors alone) and
# hit@5; keyword matching still matters for exact reference numbers. Small sample: revisit
# as the golden set grows.
KEYWORD_WEIGHT = 0.3
CANDIDATES = 50  # per method, before fusion
CHUNK_RANK_NORMALIZATION = 5  # as for documents: log length + proximity (best on golden set)


@dataclass
class ChunkHit:
    chunk_id: int
    doc_id: int
    source_id: str | None
    title: str
    heading: str | None
    text: str
    char_start: int
    char_end: int
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None


def _filters(f: SearchFilters | None) -> list:
    f = f or SearchFilters()
    clauses = []
    if f.as_of is not None:
        clauses.append(Chunk.valid_from <= f.as_of)
        clauses.append(or_(Chunk.valid_to.is_(None), Chunk.valid_to > f.as_of))
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


def keyword_candidates(
    session: Session,
    query: str,
    n: int = CANDIDATES,
    filters: SearchFilters | None = None,
    mode: Mode = "any",
    normalization: int = CHUNK_RANK_NORMALIZATION,
) -> list[int]:
    tsq = build_tsquery(query, mode)
    if session.scalar(select(func.numnode(tsq))) == 0:
        return []
    score = func.ts_rank_cd(Chunk.tsv, tsq, literal(normalization))
    rows = session.scalars(
        select(Chunk.id)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.tsv.op("@@")(tsq), and_(*_filters(filters)))
        .order_by(score.desc(), Chunk.id)
        .limit(n)
    )
    return list(rows)


def vector_candidates(
    session: Session,
    query_vector: list[float],
    n: int = CANDIDATES,
    filters: SearchFilters | None = None,
) -> list[int]:
    # HNSW filters after the graph search; widen the search so filters still leave n rows.
    session.execute(text(f"SET LOCAL hnsw.ef_search = {max(100, n * 4)}"))
    rows = session.scalars(
        select(Chunk.id)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.embedding.is_not(None), and_(*_filters(filters)))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(n)
    )
    return list(rows)


def rrf(
    ranked_lists: list[list[int]], k: int = RRF_K, weights: list[float] | None = None
) -> dict[int, float]:
    """Reciprocal rank fusion: sum of weight / (k + rank) over the lists an item appears in."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return scores


def search_chunks(
    session: Session,
    query: str,
    embedder: Embedder | None = None,
    k: int = 10,
    filters: SearchFilters | None = None,
    method: Method = "hybrid",
    candidates: int = CANDIDATES,
    keyword_weight: float = KEYWORD_WEIGHT,
) -> list[ChunkHit]:
    kw = keyword_candidates(session, query, candidates, filters) if method != "vector" else []
    vec: list[int] = []
    if method != "keyword":
        if embedder is None:
            raise ValueError(f"method {method!r} needs an embedder")
        vec = vector_candidates(session, embedder.embed_query(query), candidates, filters)
    scores = rrf([kw, vec], weights=[keyword_weight, 1.0])
    top = sorted(scores, key=lambda cid: -scores[cid])[:k]
    if not top:
        return []
    kw_rank = {cid: r for r, cid in enumerate(kw, start=1)}
    vec_rank = {cid: r for r, cid in enumerate(vec, start=1)}
    rows = {
        c.id: (c, d)
        for c, d in session.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.doc_id)
            .where(Chunk.id.in_(top))
        ).all()
    }
    return [
        ChunkHit(
            chunk_id=cid,
            doc_id=rows[cid][1].id,
            source_id=rows[cid][1].source_id,
            title=rows[cid][1].title,
            heading=rows[cid][0].section_ref,
            text=rows[cid][0].text,
            char_start=rows[cid][0].char_start,
            char_end=rows[cid][0].char_end,
            score=scores[cid],
            keyword_rank=kw_rank.get(cid),
            vector_rank=vec_rank.get(cid),
        )
        for cid in top
    ]


def rank_documents(hits: list[ChunkHit]) -> list[str]:
    """Documents in order of their best chunk (hits are already sorted by score)."""
    seen: list[str] = []
    for h in hits:
        if h.source_id and h.source_id not in seen:
            seen.append(h.source_id)
    return seen
