"""Follows and alerts against the real Postgres (skipped when unreachable). Rolled back."""

from datetime import UTC, date, datetime, timedelta

import pytest
from fakes import add_doc
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.bot.alerts import follow, following, mark_sent, parse_target, pending_alerts, unfollow
from niyam.db.models import Relation, Subscription
from niyam.db.session import get_engine

CHAT = 999_000_111


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


def mine(alerts):
    return [a for a in alerts if a.chat_id == CHAT]


def test_parse_target():
    assert parse_target("13156") == ("doc", "13156")
    assert parse_target("  Digital   Lending ") == ("topic", "digital lending")


def test_follow_unfollow_and_list(session):
    add_doc(session, "f-1", "Zorblax Directions", "x")
    sub, new = follow(session, CHAT, "zorblax frimble")
    again, new_again = follow(session, CHAT, "Zorblax  Frimble")
    assert new and not new_again and again.id == sub.id
    with pytest.raises(ValueError):
        follow(session, CHAT, "424242")  # no such document
    with pytest.raises(ValueError):
        follow(session, CHAT, "ab")
    assert [s.target for s in following(session, CHAT)] == ["zorblax frimble"]
    assert unfollow(session, CHAT, "zorblax frimble") == 1
    assert following(session, CHAT) == []


def test_document_follow_alerts_once_for_new_amendments(session):
    base = add_doc(session, "f-base", "Zorblax Directions", "x", issued=date(2025, 11, 28))
    # Test source ids aren't numeric, so create the document follow directly.
    sub = Subscription(chat_id=CHAT, kind="doc", target="f-base")
    session.add(sub)
    session.flush()
    old = add_doc(session, "f-old", "Zorblax Amendment 1", "x", issued=date(2026, 1, 1))
    old.created_at = sub.created_at - timedelta(days=1)  # arrived before the follow
    new = add_doc(session, "f-new", "Zorblax Amendment 2", "x", issued=date(2026, 9, 1))
    new.created_at = datetime.now(UTC) + timedelta(seconds=5)
    for src in (old, new):
        session.add(Relation(src_doc_id=src.id, dst_doc_id=base.id, type="amends", method="link"))
    session.flush()

    alerts = mine(pending_alerts(session))
    assert [a.document_id for a in alerts] == [new.id]
    assert "Zorblax Amendment 2" in alerts[0].text and "amends" in alerts[0].text
    mark_sent(session, alerts)
    assert mine(pending_alerts(session)) == []  # never repeated


def test_topic_follow_alerts_on_matching_titles(session):
    sub, _ = follow(session, CHAT, "zorblax lending")
    match = add_doc(session, "f-m", "Zorblax Lending Amendment Directions", "x")
    miss = add_doc(session, "f-x", "Zorblax Deposits", "x")
    for d in (match, miss):
        d.created_at = datetime.now(UTC) + timedelta(seconds=5)
    session.flush()
    alerts = mine(pending_alerts(session))
    assert [a.document_id for a in alerts] == [match.id]
    assert "zorblax lending" in alerts[0].text


def test_send_pending_alerts_records_only_delivered(session, monkeypatch):
    import httpx

    from niyam.bot.send import send_pending_alerts

    follow(session, CHAT, "zorblax lending")
    for sid in ("f-a", "f-b"):
        d = add_doc(session, sid, f"Zorblax Lending {sid}", "x")
        d.created_at = datetime.now(UTC) + timedelta(seconds=5)
    session.flush()
    sent_texts = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        sent_texts.append(body)
        return httpx.Response(500 if "f-b" in body else 200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    # Real subscriptions in a dev database would also be sent: keep only ours pending.
    monkeypatch.setattr("niyam.bot.send.pending_alerts", lambda s: mine(pending_alerts(s)))
    assert send_pending_alerts(session, "TOKEN", client) == 1
    assert len(mine(pending_alerts(session))) == 1  # the failed one is retried next time
    assert all('"parse_mode": "HTML"' in t or '"parse_mode":"HTML"' in t for t in sent_texts)
