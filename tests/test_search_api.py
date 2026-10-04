from datetime import date

import pytest
from fastapi.testclient import TestClient

from niyam.api import main
from niyam.db.session import get_session
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
