"""Chunking and embedding into the index, against the real Postgres (skipped when
unreachable). Test documents use made-up words; everything is rolled back."""

from datetime import date

import pytest
from fakes import FakeEmbedder, add_doc, index_document
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.models import EMBEDDING_DIM, Chunk
from niyam.db.session import get_engine
from niyam.ingest.index import build_index, chunk_context, embed_missing


@pytest.fixture
def session():
    try:
        conn = get_engine().connect()
    except OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d db`")
    trans = conn.begin()
    with Session(bind=conn, join_transaction_mode="create_savepoint") as s:
        yield s
    trans.rollback()
    conn.close()


def test_chunk_context():
    assert chunk_context("Title", None) == "Title"
    assert chunk_context("Title", "Chapter I > 1. Scope") == "Title\nChapter I > 1. Scope"


def test_index_document_stores_chunks_with_offsets_context_and_validity(session):
    text = "Chapter I – Zorblax\n\n1. Frimble rules\n\n" + " ".join(["quantilox"] * 60)
    d = add_doc(session, "h-1", "Zorblax Directions", text, valid_to=date(2030, 1, 1))
    n = index_document(session, d, FakeEmbedder())
    session.flush()
    chunks = session.scalars(select(Chunk).where(Chunk.doc_id == d.id)).all()
    assert n == len(chunks) == 1
    c = chunks[0]
    assert text[c.char_start : c.char_end] == c.text
    assert c.section_ref == "Chapter I – Zorblax > 1. Frimble rules"
    assert c.context.startswith("Zorblax Directions\n")
    assert (c.valid_from, c.valid_to) == (date(2024, 1, 1), date(2030, 1, 1))
    assert len(c.embedding) == EMBEDDING_DIM


def test_build_index_skips_documents_that_already_have_chunks(session):
    add_doc(session, "h-1", "Zorblax one", " ".join(["zorblax"] * 50))
    add_doc(session, "h-2", "Zorblax two", " ".join(["frimble"] * 50))
    only = ["h-1", "h-2"]  # leave real documents in a dev database alone
    first = build_index(session, FakeEmbedder(), source_ids=only)
    second = build_index(session, FakeEmbedder(), source_ids=only)
    assert (first.documents, second.documents) == (2, 0)
    assert first.chunks == first.embedded == 2


def test_chunk_only_then_embed_later(session):
    d = add_doc(session, "h-1", "Zorblax one", " ".join(["zorblax"] * 50))
    stats = build_index(session, None, source_ids=["h-1"])
    assert (stats.chunks, stats.embedded) == (1, 0)
    (c,) = session.scalars(select(Chunk).where(Chunk.doc_id == d.id)).all()
    assert c.embedding is None
    assert embed_missing(session, FakeEmbedder(), doc_ids=[d.id]) == 1
    session.refresh(c)
    assert len(c.embedding) == EMBEDDING_DIM
