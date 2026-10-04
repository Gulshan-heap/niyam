"""Answer a question: retrieve passages, ask the model for quoted sentences, verify quotes."""

import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from niyam.db.models import Document
from niyam.rag.llm import LLM
from niyam.rag.prompt import build_messages, parse_reply
from niyam.rag.verify import PassageRef, Sentence, verify_sentences
from niyam.retrieval.embeddings import Embedder
from niyam.retrieval.hybrid import ChunkHit, search_chunks
from niyam.retrieval.keyword import SearchFilters
from niyam.retrieval.temporal import expand_with_amendments

log = logging.getLogger(__name__)

PASSAGES = 8


@dataclass
class Answer:
    question: str
    as_of: date | None
    abstained: bool
    sentences: list[Sentence]
    passages: list[PassageRef]
    note: str | None = None  # why we abstained, when we did
    unsupported: list[str] = field(default_factory=list)  # sentences dropped as unverified

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


def to_passages(session: Session, hits: list[ChunkHit], as_of: date | None) -> list[PassageRef]:
    docs = {
        d.id: d
        for d in session.scalars(select(Document).where(Document.id.in_([h.doc_id for h in hits])))
    }
    on = as_of or date.today()
    out = []
    for i, h in enumerate(hits, start=1):
        d = docs[h.doc_id]
        in_force = (d.valid_from is None or d.valid_from <= on) and (
            d.valid_to is None or d.valid_to > on
        )
        out.append(
            PassageRef(
                label=f"P{i}",
                chunk_id=h.chunk_id,
                source_id=h.source_id,
                title=h.title,
                heading=h.heading,
                text=h.text,
                char_start=h.char_start,
                url=d.url,
                issued_date=d.issued_date.isoformat() if d.issued_date else None,
                in_force=in_force,
            )
        )
    return out


def answer_question(
    session: Session,
    question: str,
    llm: LLM,
    embedder: Embedder,
    as_of: date | None = None,
    k: int = PASSAGES,
) -> Answer:
    """Answer with the rules in force on `as_of` (today when not given)."""
    as_of = as_of or date.today()
    hits = search_chunks(session, question, embedder, k=k, filters=SearchFilters(as_of=as_of))
    hits = expand_with_amendments(session, question, hits, as_of)
    passages = to_passages(session, hits, as_of)
    if not passages:
        return Answer(question, as_of, True, [], [], note="No matching passages were found.")

    reply = llm.complete(build_messages(question, passages, as_of.isoformat() if as_of else None))
    try:
        data = parse_reply(reply)
    except ValueError as exc:
        log.warning("unparseable model reply: %s", exc)
        return Answer(question, as_of, True, [], passages, note="The model reply was not valid.")

    if data.get("abstained"):
        note = " ".join(str(s.get("text", "")) for s in data["sentences"]) or None
        return Answer(question, as_of, True, [], passages, note=note)

    sentences = verify_sentences(data["sentences"], passages)
    supported = [s for s in sentences if s.supported]
    dropped = [s.text for s in sentences if not s.supported]
    if not supported:
        return Answer(
            question,
            as_of,
            True,
            [],
            passages,
            note="None of the answer could be matched to the source text.",
            unsupported=dropped,
        )
    return Answer(question, as_of, False, supported, passages, unsupported=dropped)
