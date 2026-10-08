"""Chunk retrieval (keyword, vector, hybrid) against the real Postgres (skipped when
unreachable). Test documents use made-up words; everything is rolled back."""

from datetime import date

import pytest
from fakes import FakeEmbedder, add_doc, index_document
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.session import get_engine
from niyam.retrieval.hybrid import rank_documents, rrf, search_chunks
from niyam.retrieval.keyword import SearchFilters


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


def test_rrf_rewards_agreement():
    scores = rrf([[1, 2, 3], [3, 4]], k=60)
    assert max(scores, key=scores.get) == 3  # ranked by both
    assert scores[1] == pytest.approx(1 / 61)


def test_keyword_vector_and_hybrid_find_the_right_chunk(session):
    e = FakeEmbedder()
    for sid, words in [("h-a", "zorblax frimble"), ("h-b", "quantilox wibble"), ("h-c", "glorp")]:
        d = add_doc(session, sid, f"Heading {sid}", " ".join([words] * 30))
        index_document(session, d, e)
    session.flush()
    for method in ("keyword", "vector", "hybrid"):
        hits = search_chunks(session, "zorblax frimble", e, k=3, method=method)
        assert hits[0].source_id == "h-a", method
    hit = search_chunks(session, "zorblax frimble", e, k=1, method="hybrid")[0]
    assert hit.keyword_rank == 1 and hit.vector_rank == 1


def test_as_of_filter_applies_to_chunks(session):
    e = FakeEmbedder()
    old = add_doc(
        session, "h-old", "Zorblax 2016", "zorblax " * 50, issued=date(2016, 1, 1),
        valid_to=date(2025, 12, 4),
    )  # fmt: skip
    new = add_doc(session, "h-new", "Zorblax 2025", "zorblax " * 50, issued=date(2025, 11, 28))
    for d in (old, new):
        index_document(session, d, e)
    session.flush()

    def docs(as_of):
        hits = search_chunks(session, "zorblax", e, k=5, filters=SearchFilters(as_of=as_of))
        return [s for s in rank_documents(hits) if s.startswith("h-")]

    assert docs(date(2020, 1, 1)) == ["h-old"]
    assert docs(date(2026, 1, 1)) == ["h-new"]


def test_vector_method_needs_an_embedder(session):
    with pytest.raises(ValueError):
        search_chunks(session, "zorblax", None, method="hybrid")


def test_rrf_weights_shift_the_balance():
    keyword, vector = [1, 2], [2, 1]
    assert rrf([keyword, vector]) == pytest.approx({1: 1 / 61 + 1 / 62, 2: 1 / 62 + 1 / 61})
    weighted = rrf([keyword, vector], weights=[0.3, 1.0])
    assert max(weighted, key=weighted.get) == 2  # the vector list's top item wins
