"""Runs against the real Postgres from `docker compose up db` (after `alembic upgrade head`).

Skipped automatically when the database is unreachable. Each test runs in a transaction
that is rolled back, so the database is left untouched.
"""

from datetime import date

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.models import EMBEDDING_DIM, Chunk, Document
from niyam.db.session import get_engine


@pytest.fixture
def session():
    try:
        conn = get_engine().connect()
    except OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d db`")
    trans = conn.begin()
    with Session(bind=conn) as s:
        yield s
    trans.rollback()
    conn.close()


def _doc(s: Session, circular_no: str, valid_from: date, valid_to: date | None) -> Document:
    d = Document(
        regulator="RBI",
        circular_no=circular_no,
        title=f"Test {circular_no}",
        doc_type="circular",
        issued_date=valid_from,
        valid_from=valid_from,
        valid_to=valid_to,
        url=f"https://example.test/{circular_no}",
    )
    s.add(d)
    s.flush()
    return d


def _chunk(s: Session, doc: Document, body: str, vec: list[float]) -> Chunk:
    c = Chunk(
        doc_id=doc.id,
        ord=0,
        text=body,
        char_start=0,
        char_end=len(body),
        embedding=vec,
        valid_from=doc.valid_from,
        valid_to=doc.valid_to,
    )
    s.add(c)
    s.flush()
    return c


def test_tsvector_is_generated_and_searchable(session):
    d = _doc(session, "T/1", date(2020, 1, 1), None)
    _chunk(session, d, "Periodic updation of KYC for high risk customers", [0.0] * EMBEDDING_DIM)
    hit = session.execute(
        select(Chunk.id).where(
            Chunk.tsv.op("@@")(text("plainto_tsquery('english', 'kyc updation')"))
        )
    ).scalar()
    assert hit is not None


def test_vector_cosine_search_orders_by_similarity(session):
    d = _doc(session, "T/2", date(2020, 1, 1), None)
    near = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    far = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    c_near = _chunk(session, d, "near", near)
    c_far = Chunk(doc_id=d.id, ord=1, text="far", char_start=0, char_end=3, embedding=far)
    session.add(c_far)
    session.flush()
    ids = (
        session.execute(
            select(Chunk.id)
            .where(Chunk.doc_id == d.id)
            .order_by(Chunk.embedding.cosine_distance(near))
        )
        .scalars()
        .all()
    )
    assert ids == [c_near.id, c_far.id]


def test_validity_window_filter(session):
    old = _doc(session, "T/old", date(2015, 1, 1), date(2021, 6, 1))
    new = _doc(session, "T/new", date(2021, 6, 1), None)
    vec = [0.0] * EMBEDDING_DIM
    _chunk(session, old, "old rule", vec)
    _chunk(session, new, "new rule", vec)

    def in_force(as_of: date) -> set[str]:
        rows = session.execute(
            select(Chunk.text).where(
                Chunk.doc_id.in_([old.id, new.id]),
                Chunk.valid_from <= as_of,
                (Chunk.valid_to.is_(None)) | (Chunk.valid_to > as_of),
            )
        ).scalars()
        return set(rows)

    assert in_force(date(2018, 3, 1)) == {"old rule"}
    assert in_force(date(2021, 6, 1)) == {"new rule"}  # boundary: superseded on that day
    assert in_force(date(2026, 1, 1)) == {"new rule"}
