"""Temporal retrieval helpers: pull in the amendments of what was retrieved.

A Master Direction's text can be older than the rule in force: amendment directions change
it afterwards (and PDF-only Directions are held as an older dated version). When a
retrieved passage comes from a document that was later amended, the amending documents in
force on the question's date are added to the context, each by its best-matching passage,
so the answer can reflect the latest change.
"""

from datetime import date

from sqlalchemy import func, literal, or_, select
from sqlalchemy.orm import Session, aliased

from niyam.db.models import Chunk, Document, Relation
from niyam.retrieval.hybrid import ChunkHit
from niyam.retrieval.keyword import build_tsquery

MAX_AMENDMENTS = 3


def amendments_in_force(session: Session, doc_ids: list[int], as_of: date) -> list[int]:
    """Documents that amend any of `doc_ids`, issued on or before `as_of` and in force then,
    newest first."""
    if not doc_ids:
        return []
    amender = aliased(Document)
    rows = session.execute(
        select(amender.id)
        .join(Relation, Relation.src_doc_id == amender.id)
        .where(
            Relation.type == "amends",
            Relation.dst_doc_id.in_(doc_ids),
            amender.issued_date <= as_of,
            or_(amender.valid_to.is_(None), amender.valid_to > as_of),
            or_(as_of < date.today(), amender.is_withdrawn.is_(False)),
        )
        .group_by(amender.id, amender.issued_date)
        .order_by(amender.issued_date.desc())
    ).scalars()
    return [i for i in rows if i not in doc_ids]


def best_chunk(session: Session, doc_id: int, query: str) -> Chunk | None:
    """The document's chunk that best matches the query (its first chunk if none match)."""
    tsq = build_tsquery(query)
    score = func.ts_rank_cd(Chunk.tsv, tsq, literal(5))
    return session.scalars(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(score.desc(), Chunk.ord).limit(1)
    ).first()


def expand_with_amendments(
    session: Session,
    query: str,
    hits: list[ChunkHit],
    as_of: date,
    limit: int = MAX_AMENDMENTS,
) -> list[ChunkHit]:
    """`hits` plus the best passage of up to `limit` amendments of the retrieved documents."""
    doc_ids = list(dict.fromkeys(h.doc_id for h in hits))
    extra: list[ChunkHit] = []
    for amender_id in amendments_in_force(session, doc_ids, as_of)[:limit]:
        chunk = best_chunk(session, amender_id, query)
        if chunk is None:
            continue
        doc = session.get(Document, amender_id)
        extra.append(
            ChunkHit(
                chunk_id=chunk.id,
                doc_id=doc.id,
                source_id=doc.source_id,
                title=doc.title,
                heading=chunk.section_ref,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                score=0.0,  # added for context, not ranked
            )
        )
    return hits + extra
