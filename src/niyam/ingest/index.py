"""Build the retrieval index in two resumable steps.

1. `chunk_documents`: split every document that has no chunks yet (seconds for the corpus).
2. `embed_missing`: embed chunks whose embedding is still NULL, in batches with a commit
   after each, so a long run (hours on a laptop CPU) can be stopped and resumed. Swapping
   the embedding model only needs `UPDATE chunks SET embedding = NULL` and a re-run.

The ingestion pipeline drops a document's chunks when its text changes, so changed
documents are re-chunked and re-embedded on the next run.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from niyam.db.models import Chunk, Document
from niyam.ingest.chunk import chunk_text
from niyam.retrieval.embeddings import Embedder

log = logging.getLogger(__name__)

EMBED_BATCH = 64


@dataclass
class IndexStats:
    documents: int = 0
    chunks: int = 0
    embedded: int = 0


def chunk_context(title: str, heading: str | None) -> str:
    return f"{title}\n{heading}" if heading else title


def embedding_input(chunk: Chunk) -> str:
    """What the embedding model reads: title and heading for context, then the passage."""
    return f"{chunk.context}\n\n{chunk.text}" if chunk.context else chunk.text


def chunk_document(session: Session, doc: Document) -> int:
    spans = chunk_text(doc.raw_text or "")
    session.add_all(
        Chunk(
            doc_id=doc.id,
            ord=s.ord,
            text=s.text,
            section_ref=s.heading,
            context=chunk_context(doc.title, s.heading),
            char_start=s.char_start,
            char_end=s.char_end,
            valid_from=doc.valid_from,
            valid_to=doc.valid_to,
        )
        for s in spans
    )
    return len(spans)


def chunk_documents(
    session: Session, limit: int | None = None, source_ids: list[str] | None = None
) -> IndexStats:
    """Chunk documents that have text but no chunks yet (optionally only `source_ids`)."""
    has_chunks = exists().where(Chunk.doc_id == Document.id)
    query = select(Document).where(Document.raw_text.is_not(None), ~has_chunks)
    if source_ids is not None:
        query = query.where(Document.source_id.in_(source_ids))
    stats = IndexStats()
    for doc in session.scalars(query.order_by(Document.issued_date.desc()).limit(limit)).all():
        stats.chunks += chunk_document(session, doc)
        stats.documents += 1
    session.commit()
    log.info("chunked %d documents into %d chunks", stats.documents, stats.chunks)
    return stats


def embed_missing(
    session: Session,
    embedder: Embedder,
    batch_size: int = EMBED_BATCH,
    limit: int | None = None,
    doc_ids: list[int] | None = None,
) -> int:
    """Embed chunks that have no embedding yet, newest documents first. Returns the count."""
    base = select(Chunk).where(Chunk.embedding.is_(None))
    if doc_ids is not None:
        base = base.where(Chunk.doc_id.in_(doc_ids))
    total = session.scalar(select(func.count()).select_from(base.subquery()))
    if limit is not None:
        total = min(total, limit)
    done, started = 0, time.monotonic()
    while done < total:
        batch = session.scalars(
            base.join(Document, Document.id == Chunk.doc_id)
            .order_by(Document.issued_date.desc(), Chunk.doc_id, Chunk.ord)
            .limit(min(batch_size, total - done))
        ).all()
        if not batch:
            break
        vectors = embedder.embed_passages([embedding_input(c) for c in batch])
        for c, v in zip(batch, vectors, strict=True):
            c.embedding = v
        session.commit()
        done += len(batch)
        rate = done / max(time.monotonic() - started, 1e-9)
        log.info(
            "embedded %d/%d chunks (%.1f/s, ~%.0f min left)",
            done,
            total,
            rate,
            (total - done) / rate / 60,
        )
    return done


def build_index(
    session: Session,
    embedder: Embedder | None,
    limit: int | None = None,
    source_ids: list[str] | None = None,
) -> IndexStats:
    """Chunk new documents, then embed their chunks (skipped when embedder is None)."""
    stats = chunk_documents(session, limit=limit, source_ids=source_ids)
    if embedder is not None:
        doc_ids = None
        if source_ids is not None:
            doc_ids = list(
                session.scalars(select(Document.id).where(Document.source_id.in_(source_ids)))
            )
        stats.embedded = embed_missing(session, embedder, doc_ids=doc_ids)
    return stats
