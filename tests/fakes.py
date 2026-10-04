"""Test doubles shared by the indexing and retrieval tests."""

import hashlib
import math
from datetime import date

from sqlalchemy.orm import Session

from niyam.db.models import EMBEDDING_DIM, Document
from niyam.ingest.index import chunk_document, embed_missing


class FakeEmbedder:
    """Hashes words into a normalised 384-d bag-of-words vector: texts sharing words are close."""

    dim = EMBEDDING_DIM

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def add_doc(s: Session, sid: str, title: str, text: str, **kw) -> Document:
    issued = kw.pop("issued", date(2024, 1, 1))
    d = Document(
        regulator="RBI",
        source_id=sid,
        title=title,
        raw_text=text,
        doc_type="circular",
        issued_date=issued,
        valid_from=kw.pop("valid_from", issued),
        url=f"https://example.test/{sid}",
        **kw,
    )
    s.add(d)
    s.flush()
    return d


def index_document(session: Session, doc: Document, embedder) -> int:
    n = chunk_document(session, doc)
    session.flush()
    embed_missing(session, embedder, doc_ids=[doc.id])
    return n
