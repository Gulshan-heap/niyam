"""The LangGraph agent end to end against the real Postgres (skipped when unreachable),
with a fake embedder and scripted fake LLMs for grading and answering."""

import json
from datetime import date

import pytest
from fakes import FakeEmbedder, add_doc, index_document
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.agent.graph import run_agent
from niyam.db.session import get_engine

TEXT = (
    "Chapter III – Zorblax lending\n\n11. Frimble period\n\n(1) The borrower shall be given an "
    "explicit option to exit a zorblax loan without any penalty during an initial frimble "
    "period, subject to the period so determined not being less than one day."
)
ANSWER = {
    "abstained": False,
    "sentences": [
        {
            "text": "The frimble period is at least one day.",
            "citations": [{"passage": "P1", "quote": "not being less than one day"}],
        }
    ],
}


class Script:
    """Returns queued replies in order and records the prompts it was given."""

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


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
def indexed(session):
    d = add_doc(session, "g-1", "Zorblax Lending Directions", TEXT, issued=date(2024, 1, 1))
    index_document(session, d, FakeEmbedder())
    session.flush()
    return d


def label_of(state, source_id="g-1"):
    return next(p.label for p in state["passages"] if p.source_id == source_id)


def test_answerable_question_goes_straight_to_a_verified_answer(session, indexed, monkeypatch):
    # Restrict retrieval to the test document so real data can't change passage labels.
    monkeypatch.setattr("niyam.agent.graph.PASSAGES", 1)
    grader = Script({"relevant": ["P1"], "answerable": True, "better_query": ""})
    llm = Script(ANSWER)
    st = run_agent(session, "zorblax frimble period exit loan", llm, FakeEmbedder(), grader=grader)
    a = st["answer"]
    assert not a.abstained and a.passages[0].source_id == "g-1"
    assert a.sentences[0].citations[0].verified
    assert st["as_of_source"] == "today"
    assert len(grader.calls) == 1 and len(llm.calls) == 1


def test_unanswerable_rewrites_then_abstains(session, indexed):
    grader = Script(
        {"relevant": [], "answerable": False, "better_query": "zorblax frimble"},
        {"relevant": [], "answerable": False, "better_query": "zorblax loan"},
        {"relevant": [], "answerable": False, "better_query": "zorblax again"},
    )
    llm = Script()  # must never be called
    st = run_agent(session, "something unclear", llm, FakeEmbedder(), grader=grader)
    assert st["answer"].abstained
    assert st["rewrites"] == 2 and len(grader.calls) == 3
    assert llm.calls == []
    assert [t for t in st["trace"] if t.startswith("rewrite")] == [
        "rewrite -> 'zorblax frimble'",
        "rewrite -> 'zorblax loan'",
    ]


def test_out_of_scope_abstains_without_rewriting(session, indexed):
    grader = Script({"relevant": [], "answerable": False, "better_query": ""})
    st = run_agent(
        session, "What is the capital of France?", Script(), FakeEmbedder(), grader=grader
    )
    assert st["answer"].abstained and st["rewrites"] == 0


def test_date_is_read_from_the_question(session, indexed):
    grader = Script({"relevant": [], "answerable": False, "better_query": ""})
    st = run_agent(session, "zorblax rule in March 2024", Script(), FakeEmbedder(), grader=grader)
    assert (st["as_of"], st["as_of_source"]) == (date(2024, 3, 15), "question")
    explicit = run_agent(
        session,
        "zorblax rule in March 2024",
        Script(),
        FakeEmbedder(),
        as_of=date(2025, 1, 1),
        grader=Script({"relevant": [], "answerable": False, "better_query": ""}),
    )
    assert (explicit["as_of"], explicit["as_of_source"]) == (date(2025, 1, 1), "request")


def test_garbled_grader_reply_keeps_passages_and_answers(session, indexed, monkeypatch):
    monkeypatch.setattr("niyam.agent.graph.PASSAGES", 1)
    st = run_agent(
        session, "zorblax frimble period", Script(ANSWER), FakeEmbedder(), grader=Script("oops")
    )
    assert not st["answer"].abstained
