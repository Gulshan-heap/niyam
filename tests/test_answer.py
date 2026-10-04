"""answer_question end to end against the real Postgres (skipped when unreachable), with a
fake embedder and a scripted fake LLM. Everything is rolled back."""

import json

import pytest
from fakes import FakeEmbedder, add_doc, index_document
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.session import get_engine
from niyam.rag.answer import answer_question

TEXT = (
    "Chapter III – Zorblax lending\n\n11. Frimble period\n\n(1) The borrower shall be given an "
    "explicit option to exit a zorblax loan without any penalty during an initial frimble "
    "period, subject to the period so determined not being less than one day."
)


class ScriptedLLM:
    def __init__(self, reply: dict | str):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.reply


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
    d = add_doc(session, "a-1", "Zorblax Lending Directions", TEXT)
    index_document(session, d, FakeEmbedder())
    session.flush()
    return d


def ask(session, llm):
    return answer_question(session, "zorblax loan frimble period exit", llm, FakeEmbedder())


def test_supported_answer_with_highlight_offsets(session, indexed):
    llm = ScriptedLLM(
        {
            "abstained": False,
            "sentences": [
                {
                    "text": "You can exit a zorblax loan without penalty for at least a day.",
                    "citations": [{"passage": "P1", "quote": "not being less than one day"}],
                }
            ],
        }
    )
    a = ask(session, llm)
    assert not a.abstained
    assert a.passages[0].source_id == "a-1"
    c = a.sentences[0].citations[0]
    assert TEXT[c.doc_char_start : c.doc_char_end] == "not being less than one day"
    assert "[P1] Zorblax Lending Directions" in llm.messages[1]["content"]


def test_unverifiable_sentences_are_dropped_and_all_unverified_abstains(session, indexed):
    mixed = ScriptedLLM(
        {
            "sentences": [
                {
                    "text": "Supported.",
                    "citations": [{"passage": "P1", "quote": "not being less than one day"}],
                },
                {
                    "text": "Invented.",
                    "citations": [
                        {"passage": "P1", "quote": "a refund within thirty days is guaranteed"}
                    ],
                },
            ]
        }
    )
    a = ask(session, mixed)
    assert not a.abstained
    assert [s.text for s in a.sentences] == ["Supported."]
    assert a.unsupported == ["Invented."]

    invented = ScriptedLLM(
        {
            "sentences": [
                {
                    "text": "Invented.",
                    "citations": [
                        {"passage": "P1", "quote": "thirty days refund guaranteed always"}
                    ],
                }
            ]
        }
    )
    assert ask(session, invented).abstained


def test_model_abstains_or_replies_garbage(session, indexed):
    a = ask(session, ScriptedLLM({"abstained": True, "sentences": [{"text": "Not covered."}]}))
    assert a.abstained and a.note == "Not covered."
    assert ask(session, ScriptedLLM("sorry, no JSON here")).abstained


def test_no_passages_means_no_llm_call(session, monkeypatch):
    monkeypatch.setattr("niyam.rag.answer.search_chunks", lambda *args, **kwargs: [])
    llm = ScriptedLLM({"sentences": []})
    a = answer_question(session, "anything", llm, FakeEmbedder())
    assert a.abstained and a.passages == []
    assert llm.messages is None
