"""Database schema.

Validity windows (valid_from / valid_to) are the core of Niyam: a document or chunk is
"in force" on date D when valid_from <= D and (valid_to is NULL or valid_to > D).
Chunks copy their document's window so retrieval can filter without a join.
"""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("regulator", "url"),
        UniqueConstraint("regulator", "source_id"),
        CheckConstraint("regulator IN ('RBI', 'SEBI')"),
        CheckConstraint("doc_type IN ('circular', 'master_direction', 'notification')"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    regulator: Mapped[str] = mapped_column(String(16))
    source_id: Mapped[str | None] = mapped_column(String(32))  # regulator's own id (RBI page Id)
    circular_no: Mapped[str | None] = mapped_column(String(128), index=True)
    rbi_no: Mapped[str | None] = mapped_column(String(64))  # e.g. RBI/2026-27/278
    title: Mapped[str] = mapped_column(Text)
    department: Mapped[str | None] = mapped_column(String(256))
    doc_type: Mapped[str] = mapped_column(String(32))
    issued_date: Mapped[date] = mapped_column(Date, index=True)
    effective_from: Mapped[date | None] = mapped_column(Date)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    # Stated by the regulator: last "Updated as on" stamp, and the date it was withdrawn.
    updated_on: Mapped[date | None] = mapped_column(Date)
    withdrawn_on: Mapped[date | None] = mapped_column(Date)
    url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    pdf_path: Mapped[str | None] = mapped_column(Text)
    source_path: Mapped[str | None] = mapped_column(Text)  # cached HTML, relative to data_dir
    raw_text: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Relation(Base):
    """A directed edge: src document amends/supersedes/repeals/refers to dst document."""

    __tablename__ = "relations"
    __table_args__ = (
        UniqueConstraint("src_doc_id", "dst_doc_id", "type"),
        CheckConstraint("type IN ('amends', 'supersedes', 'repeals', 'refers')"),
        CheckConstraint("method IN ('regex', 'llm', 'annex', 'manual')"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    src_doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    dst_doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(16))
    method: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "ord"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_validity", "valid_from", "valid_to"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    section_ref: Mapped[str | None] = mapped_column(String(64))
    page: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # [{"page": int, "bbox": [x0, y0, x1, y1]}, ...] used to highlight citations in the PDF.
    bboxes: Mapped[list | None] = mapped_column(JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)


class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        UniqueConstraint("chat_id", "kind", "target"),
        CheckConstraint("kind IN ('doc', 'topic')"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)  # Telegram ids exceed int32
    kind: Mapped[str] = mapped_column(String(16))
    target: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AlertSent(Base):
    __tablename__ = "alerts_sent"
    __table_args__ = (UniqueConstraint("subscription_id", "relation_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    subscription_id: Mapped[int] = mapped_column(ForeignKey("subscriptions.id", ondelete="CASCADE"))
    relation_id: Mapped[int] = mapped_column(ForeignKey("relations.id", ondelete="CASCADE"))
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class QueryCache(Base):
    __tablename__ = "query_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    query_embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    as_of: Mapped[date | None] = mapped_column(Date)
    answer_json: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    config_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (UniqueConstraint("run_id", "qid", "metric"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"))
    qid: Mapped[str] = mapped_column(String(64))
    metric: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
