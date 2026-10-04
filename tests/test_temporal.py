"""Amendment expansion and 'withdrawn means not in force today', against the real Postgres
(skipped when unreachable). Made-up words; everything is rolled back."""

from datetime import date, timedelta

import pytest
from fakes import FakeEmbedder, add_doc, index_document
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.models import Relation
from niyam.db.session import get_engine
from niyam.retrieval.hybrid import rank_documents, search_chunks
from niyam.retrieval.keyword import SearchFilters, keyword_search
from niyam.retrieval.temporal import amendments_in_force, expand_with_amendments


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


@pytest.fixture
def chain(session):
    """A base direction and two amendments (2024, 2026), all about 'zorblax'."""
    e = FakeEmbedder()
    base = add_doc(session, "t-base", "Zorblax Directions", "zorblax frimble rules " * 30,
                   issued=date(2023, 1, 1))  # fmt: skip
    a1 = add_doc(session, "t-a1", "Zorblax First Amendment", "zorblax limit raised " * 30,
                 issued=date(2024, 6, 1))  # fmt: skip
    a2 = add_doc(session, "t-a2", "Zorblax Second Amendment", "zorblax limit lowered " * 30,
                 issued=date(2026, 3, 1))  # fmt: skip
    for d in (base, a1, a2):
        index_document(session, d, e)
    for a in (a1, a2):
        session.add(Relation(src_doc_id=a.id, dst_doc_id=base.id, type="amends", method="link"))
    session.flush()
    return base, a1, a2


def test_amendments_in_force_respect_the_date(session, chain):
    base, a1, a2 = chain
    assert amendments_in_force(session, [base.id], date(2023, 6, 1)) == []
    assert amendments_in_force(session, [base.id], date(2025, 1, 1)) == [a1.id]
    assert amendments_in_force(session, [base.id], date(2026, 6, 1)) == [a2.id, a1.id]


def test_expansion_adds_amending_passages_once(session, chain):
    base, a1, a2 = chain
    hits = search_chunks(session, "zorblax frimble", FakeEmbedder(), k=1, method="keyword",
                         filters=SearchFilters(as_of=date(2025, 1, 1)))  # fmt: skip
    assert rank_documents(hits) == ["t-base"]
    expanded = expand_with_amendments(session, "zorblax frimble", hits, date(2025, 1, 1))
    assert rank_documents(expanded) == ["t-base", "t-a1"]  # a2 not yet issued in 2025


def test_withdrawn_without_date_is_not_in_force_today(session):
    add_doc(session, "t-w", "Zorblax withdrawn", "zorblax " * 20, is_withdrawn=True)
    add_doc(session, "t-ok", "Zorblax current", "zorblax " * 20)
    today = keyword_search(session, "zorblax", filters=SearchFilters(as_of=date.today()))
    past = keyword_search(
        session, "zorblax", filters=SearchFilters(as_of=date.today() - timedelta(days=30))
    )
    assert {h.source_id for h in today if h.source_id.startswith("t-")} == {"t-ok"}
    assert {h.source_id for h in past if h.source_id.startswith("t-")} == {"t-w", "t-ok"}


def test_timeline_endpoint_lists_amendments_newest_first(session, chain):
    from fastapi.testclient import TestClient

    from niyam.api.main import app
    from niyam.db.session import get_session

    app.dependency_overrides[get_session] = lambda: session
    try:
        base = TestClient(app).get("/documents/t-base/timeline").json()
        a1 = TestClient(app).get("/documents/t-a1/timeline").json()
    finally:
        app.dependency_overrides.clear()
    assert [d["source_id"] for d in base["amended_by"]] == ["t-a2", "t-a1"]
    assert [d["source_id"] for d in a1["amends"]] == ["t-base"]
    assert base["superseded_by"] == []
