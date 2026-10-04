"""Follows and change alerts, independent of any chat platform.

A chat follows either a document (kind "doc", target = RBI page id) or a topic (kind
"topic", target = words that must all appear in a new document's title). Alerts go out for
documents that arrived after the follow:

- doc: a new document amends, supersedes or repeals the followed document;
- topic: a new document's title contains every word of the topic.

Each (subscription, new document) pair is alerted once.
"""

import html
import re
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from niyam.db.models import AlertSent, Document, Relation, Subscription

CHANGE_TYPES = ("amends", "supersedes", "repeals")
VERBS = {"amends": "amends", "supersedes": "supersedes", "repeals": "repeals"}


@dataclass
class Alert:
    chat_id: int
    subscription_id: int
    document_id: int
    text: str  # Telegram HTML


def parse_target(arg: str) -> tuple[str, str]:
    """'/follow 13156' follows a document; '/follow digital lending' follows a topic."""
    arg = " ".join(arg.split())
    if re.fullmatch(r"\d{3,6}", arg):
        return "doc", arg
    return "topic", arg.lower()


def follow(session: Session, chat_id: int, arg: str) -> tuple[Subscription, bool]:
    """Returns the subscription and whether it is new. Raises ValueError for bad input."""
    kind, target = parse_target(arg)
    if not target or len(target) < 3:
        raise ValueError(
            "Tell me an RBI document id (e.g. 13156) or a topic (e.g. digital lending)."
        )
    if (
        kind == "doc"
        and session.scalar(select(Document.id).where(Document.source_id == target)) is None
    ):
        raise ValueError(f"I don't have an RBI document with id {target}.")
    existing = session.scalar(
        select(Subscription).where(
            Subscription.chat_id == chat_id,
            Subscription.kind == kind,
            Subscription.target == target,
        )
    )
    if existing:
        return existing, False
    sub = Subscription(chat_id=chat_id, kind=kind, target=target)
    session.add(sub)
    session.commit()
    return sub, True


def unfollow(session: Session, chat_id: int, arg: str) -> int:
    kind, target = parse_target(arg)
    n = session.execute(
        delete(Subscription).where(
            Subscription.chat_id == chat_id,
            Subscription.kind == kind,
            Subscription.target == target,
        )
    ).rowcount
    session.commit()
    return n


def following(session: Session, chat_id: int) -> list[Subscription]:
    return list(
        session.scalars(
            select(Subscription).where(Subscription.chat_id == chat_id).order_by(Subscription.id)
        )
    )


def _link(doc: Document) -> str:
    return f'<a href="{html.escape(doc.url)}">{html.escape(doc.title)}</a>'


def pending_alerts(session: Session) -> list[Alert]:
    already = {
        (a.subscription_id, a.document_id)
        for a in session.execute(select(AlertSent.subscription_id, AlertSent.document_id))
    }
    alerts: list[Alert] = []
    for sub in session.scalars(select(Subscription)):
        if sub.kind == "doc":
            followed = session.scalar(select(Document).where(Document.source_id == sub.target))
            if followed is None:
                continue
            rows = session.execute(
                select(Document, Relation.type)
                .join(Relation, Relation.src_doc_id == Document.id)
                .where(
                    Relation.dst_doc_id == followed.id,
                    Relation.type.in_(CHANGE_TYPES),
                    Document.created_at > sub.created_at,
                )
                .order_by(Document.issued_date)
            ).all()
            for doc, rtype in rows:
                if (sub.id, doc.id) in already:
                    continue
                text = (
                    f"📌 {_link(doc)} ({doc.issued_date:%d %b %Y}) {VERBS[rtype]} "
                    f"{_link(followed)}, which you follow."
                )
                alerts.append(Alert(sub.chat_id, sub.id, doc.id, text))
                already.add((sub.id, doc.id))
        else:
            words = sub.target.split()
            docs = session.scalars(
                select(Document)
                .where(Document.created_at > sub.created_at)
                .order_by(Document.issued_date)
            )
            for doc in docs:
                title = doc.title.lower()
                if (sub.id, doc.id) in already or not all(w in title for w in words):
                    continue
                text = (
                    f"🆕 New RBI document on “{html.escape(sub.target)}”: {_link(doc)} "
                    f"({doc.issued_date:%d %b %Y})."
                )
                alerts.append(Alert(sub.chat_id, sub.id, doc.id, text))
                already.add((sub.id, doc.id))
    return alerts


def mark_sent(session: Session, alerts: list[Alert]) -> None:
    session.add_all(
        AlertSent(subscription_id=a.subscription_id, document_id=a.document_id) for a in alerts
    )
    session.commit()
