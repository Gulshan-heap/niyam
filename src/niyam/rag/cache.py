"""Answer cache for /ask.

An answer is reusable for the same question, the same date, and the same corpus (any new
document makes cached answers stale). Lookup is exact first (normalised question), then
semantic: a cached answer for the same date and corpus whose question embedding is at least
SIMILARITY cosine-similar (a rewording of the same question).
"""

import hashlib
import re
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from niyam.db.models import Document, QueryCache

SIMILARITY = 0.95


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" ?.!")


def corpus_version(session: Session) -> str:
    count, latest = session.execute(
        select(func.count(Document.id), func.max(Document.created_at))
    ).one()
    return f"{count}:{latest.isoformat() if latest else '-'}"


def cache_key(question: str, as_of: date, version: str) -> str:
    raw = f"{normalize_question(question)}|{as_of.isoformat()}|{version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached(
    session: Session,
    question: str,
    as_of: date,
    version: str,
    query_vector: list[float] | None = None,
) -> dict | None:
    hit = session.scalar(
        select(QueryCache.answer_json).where(
            QueryCache.key_hash == cache_key(question, as_of, version)
        )
    )
    if hit is not None or query_vector is None:
        return hit
    distance = QueryCache.query_embedding.cosine_distance(query_vector)
    row = session.execute(
        select(QueryCache.answer_json, distance.label("d"))
        .where(
            QueryCache.as_of == as_of,
            QueryCache.corpus_version == version,
            QueryCache.query_embedding.is_not(None),
        )
        .order_by(distance)
        .limit(1)
    ).first()
    if row is not None and row.d <= 1 - SIMILARITY:
        return row.answer_json
    return None


def put_cached(
    session: Session,
    question: str,
    as_of: date,
    version: str,
    answer: dict,
    query_vector: list[float] | None = None,
) -> None:
    stmt = insert(QueryCache).values(
        key_hash=cache_key(question, as_of, version),
        query_embedding=query_vector,
        as_of=as_of,
        corpus_version=version,
        answer_json=answer,
    )
    session.execute(stmt.on_conflict_do_nothing(index_elements=["key_hash"]))
    session.commit()
