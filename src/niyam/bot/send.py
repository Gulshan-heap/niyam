"""Sending to Telegram over its HTTP API (used by the worker for alerts), and formatting
answers as Telegram HTML (used by the bot)."""

import html
import logging

import httpx
from sqlalchemy.orm import Session

from niyam.bot.alerts import mark_sent, pending_alerts
from niyam.rag.answer import Answer

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE = 4000  # Telegram's limit is 4096 characters


def send_message(token: str, chat_id: int, text: str, client: httpx.Client | None = None) -> bool:
    c = client or httpx.Client(timeout=30)
    try:
        r = c.post(
            API.format(token=token, method="sendMessage"),
            json={
                "chat_id": chat_id,
                "text": text[:MAX_MESSAGE],
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
            },
        )
        if r.status_code != 200:
            log.warning("telegram sendMessage %s: %s", r.status_code, r.text[:200])
        return r.status_code == 200
    finally:
        if client is None:
            c.close()


def send_pending_alerts(session: Session, token: str, client: httpx.Client | None = None) -> int:
    """Send every pending alert; only successfully sent ones are recorded."""
    sent = []
    for alert in pending_alerts(session):
        if send_message(token, alert.chat_id, alert.text, client):
            sent.append(alert)
    mark_sent(session, sent)
    return len(sent)


def format_answer(answer: Answer, as_of_source: str) -> str:
    """Answer sentences with numbered sources, each source linked with its verified quote."""
    when = {"question": "from your question", "request": "as you asked", "today": "today"}
    head = f"<i>Rules in force on {answer.as_of:%d %b %Y} ({when.get(as_of_source, '')})</i>"
    if answer.abstained:
        return f"{head}\n\n{html.escape(answer.note or 'I could not find this in RBI documents.')}"

    numbers: dict[str, int] = {}  # passage label -> source number
    lines, sources = [], []
    for s in answer.sentences:
        refs = []
        for c in s.citations:
            if not c.verified:
                continue
            if c.label not in numbers:
                numbers[c.label] = len(numbers) + 1
                p = next(p for p in answer.passages if p.label == c.label)
                quote = html.escape(c.quote if len(c.quote) < 160 else c.quote[:157] + "...")
                link = f'<a href="{html.escape(p.url)}">{html.escape(p.title)}</a>'
                sources.append(f"[{numbers[c.label]}] {link}\n    “{quote}”")
            refs.append(numbers[c.label])
        marks = "".join(f"[{n}]" for n in sorted(set(refs)))
        lines.append(f"{html.escape(s.text)} {marks}")
    return f"{head}\n\n" + "\n".join(lines) + "\n\n<b>Sources</b>\n" + "\n".join(sources)
