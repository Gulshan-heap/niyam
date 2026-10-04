from datetime import date

import pytest
from fastapi.testclient import TestClient

from niyam.api import main
from niyam.db.session import get_session
from niyam.retrieval.hybrid import ChunkHit
from niyam.retrieval.keyword import SearchHit

client = TestClient(main.app)

HIT = SearchHit(
    doc_id=1,
    source_id="13090",
    title="Reserve Bank of India (Payments Banks – Know Your Customer) Directions, 2025",
    circular_no="DOR.AML.REC.1/14.01.001/2025-26",
    doc_type="master_direction",
    department="Department of Regulation",
    issued_date=date(2025, 11, 28),
    is_withdrawn=False,
    withdrawn_on=None,
    url="https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13090&Mode=0",
    score=33.7,
    snippet="periodic «updation» of «KYC»",
)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_search(session, query, k, filters, mode):
        seen.append({"query": query, "k": k, "filters": filters, "mode": mode})
        return [HIT]

    monkeypatch.setattr(main, "keyword_search", fake_search)
    main.app.dependency_overrides[get_session] = lambda: None
    yield seen
    main.app.dependency_overrides.clear()


def test_search_returns_results(calls):
    resp = client.get("/search", params={"q": "periodic KYC updation"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["results"][0]["source_id"] == "13090"
    assert body["results"][0]["issued_date"] == "2025-11-28"
    assert calls[0]["k"] == 10 and calls[0]["mode"] == "any"


def test_search_passes_filters(calls):
    resp = client.get(
        "/search",
        params={
            "q": "kyc",
            "k": 3,
            "mode": "all",
            "as_of": "2024-03-01",
            "doc_type": "master_direction",
            "department": "regulation",
            "issued_from": "2016-01-01",
            "issued_to": "2025-12-31",
        },
    )
    assert resp.status_code == 200
    f = calls[0]["filters"]
    assert (calls[0]["k"], calls[0]["mode"]) == (3, "all")
    assert f.as_of == date(2024, 3, 1)
    assert f.doc_type == "master_direction"
    assert f.department == "regulation"
    assert (f.issued_from, f.issued_to) == (date(2016, 1, 1), date(2025, 12, 31))


@pytest.mark.parametrize(
    "params",
    [{"q": "a"}, {"q": "kyc", "k": 0}, {"q": "kyc", "k": 51}, {"q": "kyc", "doc_type": "memo"}],
)
def test_search_rejects_bad_params(calls, params):
    assert client.get("/search", params=params).status_code == 422
    assert calls == []


CHUNK = ChunkHit(
    chunk_id=7,
    doc_id=1,
    source_id="13156",
    title="Reserve Bank of India (Commercial Banks – Credit Facilities) Directions, 2025",
    heading="Chapter III - Digital Lending Guidelines > 11. Cooling-off period",
    text="(1) The borrower shall be given an explicit option to exit a digital loan ...",
    char_start=120_400,
    char_end=121_100,
    score=0.0328,
    keyword_rank=1,
    vector_rank=2,
)


@pytest.fixture
def passage_calls(monkeypatch):
    seen = []

    def fake_search_chunks(session, query, embedder, k, filters, method):
        seen.append({"query": query, "k": k, "filters": filters, "method": method})
        return [CHUNK]

    monkeypatch.setattr(main, "search_chunks", fake_search_chunks)
    main.app.dependency_overrides[get_session] = lambda: None
    main.app.dependency_overrides[main.embedder_dependency] = lambda: object()
    yield seen
    main.app.dependency_overrides.clear()


def test_passages_returns_highlightable_chunks(passage_calls):
    resp = client.get("/passages", params={"q": "can I exit a digital loan", "as_of": "2026-01-01"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "hybrid" and body["count"] == 1
    p = body["results"][0]
    assert p["heading"].endswith("11. Cooling-off period")
    assert (p["char_start"], p["char_end"]) == (120_400, 121_100)
    assert p["url"].endswith("NotificationUser.aspx?Id=13156&Mode=0")
    assert passage_calls[0]["filters"].as_of == date(2026, 1, 1)


def test_passages_method_choice(passage_calls):
    assert client.get("/passages", params={"q": "kyc", "method": "keyword"}).status_code == 200
    assert passage_calls[0]["method"] == "keyword"
    assert client.get("/passages", params={"q": "kyc", "method": "bm25"}).status_code == 422


def test_ask_returns_sentences_with_verified_citations(monkeypatch):
    from niyam.rag.answer import Answer
    from niyam.rag.verify import Citation, PassageRef, Sentence

    p = PassageRef("P1", 7, "13156", "Credit Facilities Directions", "11. Cooling-off period",
                   "...", 100, "https://example.test/13156", "2025-11-28", True)  # fmt: skip
    cite = Citation("P1", "not being less than one day", True, 100.0, 7, "13156", 180, 207)
    seen = {}

    def fake_answer(session, question, llm, embedder, as_of=None):
        seen.update(question=question, as_of=as_of)
        return Answer(question, as_of, False, [Sentence("At least one day.", [cite])], [p])

    monkeypatch.setattr(main, "answer_question", fake_answer)
    main.app.dependency_overrides[get_session] = lambda: None
    main.app.dependency_overrides[main.embedder_dependency] = lambda: object()
    main.app.dependency_overrides[main.llm_dependency] = lambda: object()
    try:
        resp = client.post("/ask", json={"question": "cooling-off period?", "as_of": "2026-01-01"})
        bad = client.post("/ask", json={"question": "x"})
    finally:
        main.app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "At least one day." and not body["abstained"]
    c = body["sentences"][0]["citations"][0]
    assert c["verified"] and (c["doc_char_start"], c["doc_char_end"]) == (180, 207)
    assert body["passages"][0]["in_force"] is True
    assert seen["as_of"] == date(2026, 1, 1)
    assert bad.status_code == 422


class FakeSession:
    def __init__(self, doc):
        self.doc = doc

    def scalar(self, _query):
        return self.doc


def test_get_document_and_404():
    from niyam.db.models import Document

    doc = Document(
        source_id="13156", title="Credit Facilities Directions", circular_no="DOR.X", rbi_no=None,
        doc_type="master_direction", department=None, issued_date=date(2025, 11, 28),
        updated_on=None, text_as_of=date(2025, 11, 28), is_withdrawn=False, withdrawn_on=None,
        valid_from=date(2025, 11, 28), valid_to=None, url="https://example.test/13156",
        pdf_url=None, raw_text="full text",
    )  # fmt: skip
    main.app.dependency_overrides[get_session] = lambda: FakeSession(doc)
    try:
        ok = client.get("/documents/13156")
        main.app.dependency_overrides[get_session] = lambda: FakeSession(None)
        missing = client.get("/documents/1")
    finally:
        main.app.dependency_overrides.clear()
    assert ok.status_code == 200 and ok.json()["raw_text"] == "full text"
    assert missing.status_code == 404
