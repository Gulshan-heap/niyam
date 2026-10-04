"""Answer cache against the real Postgres (skipped when unreachable). Rolled back."""

from datetime import date

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.session import get_engine
from niyam.rag.cache import cache_key, get_cached, normalize_question, put_cached


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


def vec(*hot: int) -> list[float]:
    v = [0.0] * 384
    for i in hot:
        v[i] = 1.0
    return v


D = date(2026, 1, 1)


def test_normalisation_and_key():
    assert normalize_question("  What is  the LIMIT? ") == "what is the limit"
    assert cache_key("What is the limit?", D, "v1") == cache_key("what is the limit", D, "v1")
    assert cache_key("q", D, "v1") != cache_key("q", D, "v2")
    assert cache_key("q", D, "v1") != cache_key("q", date(2025, 1, 1), "v1")


def test_exact_and_semantic_hits(session):
    put_cached(session, "zorblax limit?", D, "test-v1", {"answer": "ten"}, vec(1, 2))
    assert get_cached(session, "Zorblax limit", D, "test-v1") == {"answer": "ten"}
    # A rewording with a near-identical embedding hits; a different one misses.
    assert get_cached(session, "limit for zorblax?", D, "test-v1", vec(1, 2)) == {"answer": "ten"}
    assert get_cached(session, "something else", D, "test-v1", vec(7)) is None


def test_other_date_or_corpus_misses(session):
    put_cached(session, "zorblax limit?", D, "test-v1", {"answer": "ten"}, vec(1))
    assert get_cached(session, "zorblax limit?", date(2025, 1, 1), "test-v1", vec(1)) is None
    assert get_cached(session, "zorblax limit?", D, "test-v2", vec(1)) is None


def test_putting_twice_keeps_the_first(session):
    put_cached(session, "zorblax?", D, "test-v1", {"answer": "a"})
    put_cached(session, "zorblax?", D, "test-v1", {"answer": "b"})
    assert get_cached(session, "zorblax?", D, "test-v1") == {"answer": "a"}
