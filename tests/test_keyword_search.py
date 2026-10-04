"""Keyword search against the real Postgres (skipped when unreachable).

Documents use made-up words ("zorblax", "quantile") so real data in a dev database can't
match; everything is rolled back at the end.
"""

from datetime import date

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.models import Document
from niyam.db.session import get_engine
from niyam.retrieval.keyword import SearchFilters, keyword_search


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


def add(s: Session, sid: str, title: str, text: str, **kw) -> Document:
    issued = kw.pop("issued", date(2024, 1, 1))
    d = Document(
        regulator="RBI",
        source_id=sid,
        title=title,
        raw_text=text,
        doc_type=kw.pop("doc_type", "circular"),
        issued_date=issued,
        valid_from=kw.pop("valid_from", issued),
        url=f"https://example.test/{sid}",
        **kw,
    )
    s.add(d)
    s.flush()
    return d


def ids(hits) -> list[str]:
    return [h.source_id for h in hits]


def test_title_match_outranks_body_match(session):
    add(session, "t-body", "Unrelated heading", "Rules on zorblax frimble are set out below.")
    add(session, "t-title", "Zorblax Frimble Directions", "Rules are set out below.")
    assert ids(keyword_search(session, "zorblax frimble")) == ["t-title", "t-body"]


def test_any_mode_matches_partial_questions_all_mode_does_not(session):
    add(session, "t-1", "Heading", "The zorblax frimble applies.")
    question = "zorblax frimble quantilox"  # last word appears nowhere
    assert ids(keyword_search(session, question, mode="any")) == ["t-1"]
    assert keyword_search(session, question, mode="all") == []


def test_stemming(session):
    add(session, "t-1", "Heading", "Accounts shall be zorblaxed periodically.")
    assert ids(keyword_search(session, "zorblaxing")) == ["t-1"]


def test_as_of_filter_returns_rule_in_force(session):
    add(
        session,
        "t-old",
        "Zorblax Directions 2016",
        "zorblax",
        issued=date(2016, 1, 1),
        valid_to=date(2025, 12, 4),
    )
    add(session, "t-new", "Zorblax Directions 2025", "zorblax", issued=date(2025, 11, 28))

    def in_force(d: date) -> set[str]:
        return set(ids(keyword_search(session, "zorblax", filters=SearchFilters(as_of=d))))

    assert in_force(date(2020, 6, 1)) == {"t-old"}
    assert in_force(date(2025, 12, 1)) == {"t-old", "t-new"}  # overlap before withdrawal
    assert in_force(date(2026, 1, 1)) == {"t-new"}


def test_doc_type_department_and_date_filters(session):
    add(
        session,
        "t-md",
        "Zorblax MD",
        "zorblax",
        doc_type="master_direction",
        issued=date(2020, 1, 1),
    )
    add(
        session,
        "t-c",
        "Zorblax circular",
        "zorblax",
        department="Department of Regulation",
        issued=date(2024, 5, 1),
    )
    q = "zorblax"
    assert ids(keyword_search(session, q, filters=SearchFilters(doc_type="master_direction"))) == [
        "t-md"
    ]
    assert ids(keyword_search(session, q, filters=SearchFilters(department="regulation"))) == [
        "t-c"
    ]
    f = SearchFilters(issued_from=date(2024, 1, 1), issued_to=date(2024, 12, 31))
    assert ids(keyword_search(session, q, filters=f)) == ["t-c"]


def test_snippet_highlights_terms_and_k_limits(session):
    for i in range(5):
        add(session, f"t-{i}", f"Heading {i}", f"Paragraph about zorblax number {i}.")
    hits = keyword_search(session, "zorblax", k=3)
    assert len(hits) == 3
    assert "«zorblax»" in hits[0].snippet


def test_stopword_only_query_returns_nothing(session):
    add(session, "t-1", "The of and", "the of and")
    assert keyword_search(session, "the of and") == []
