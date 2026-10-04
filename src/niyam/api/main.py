from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from niyam import __version__
from niyam.db.models import Document
from niyam.db.session import get_session
from niyam.ingest.scrapers.rbi import detail_url
from niyam.rag.answer import answer_question
from niyam.rag.llm import LLM, LiteLLM
from niyam.retrieval.embeddings import Embedder, get_embedder
from niyam.retrieval.hybrid import search_chunks
from niyam.retrieval.keyword import SearchFilters, keyword_search

app = FastAPI(title="Niyam", version=__version__)


def check_database(session: Annotated[Session, Depends(get_session)]) -> dict:
    try:
        session.execute(text("SELECT 1"))
        has_vector = session.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        return {"connected": True, "pgvector": bool(has_vector)}
    except Exception as exc:  # report, don't crash, so /health stays reachable
        return {"connected": False, "error": type(exc).__name__}


@app.get("/health")
def health(response: Response, db: Annotated[dict, Depends(check_database)]) -> dict:
    ok = db["connected"] and db.get("pgvector", False)
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "version": __version__, "database": db}


class SearchResult(BaseModel):
    doc_id: int
    source_id: str | None
    title: str
    circular_no: str | None
    doc_type: str
    department: str | None
    issued_date: date
    is_withdrawn: bool
    withdrawn_on: date | None
    url: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


@app.get("/search")
def search(
    session: Annotated[Session, Depends(get_session)],
    q: Annotated[str, Query(min_length=2, max_length=500, description="Search text")],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    mode: Literal["any", "all"] = "any",
    as_of: Annotated[date | None, Query(description="Only rules in force on this date")] = None,
    doc_type: Literal["circular", "master_direction", "notification"] | None = None,
    department: str | None = None,
    issued_from: date | None = None,
    issued_to: date | None = None,
) -> SearchResponse:
    filters = SearchFilters(
        as_of=as_of,
        doc_type=doc_type,
        department=department,
        issued_from=issued_from,
        issued_to=issued_to,
    )
    hits = keyword_search(session, q, k=k, filters=filters, mode=mode)
    return SearchResponse(
        query=q,
        count=len(hits),
        results=[SearchResult(**vars(h)) for h in hits],
    )


class Passage(BaseModel):
    chunk_id: int
    source_id: str | None
    title: str
    heading: str | None
    text: str
    char_start: int
    char_end: int
    url: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class PassagesResponse(BaseModel):
    query: str
    method: str
    count: int
    results: list[Passage]


def embedder_dependency() -> Embedder:
    return get_embedder()


@app.get("/passages")
def passages(
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(embedder_dependency)],
    q: Annotated[str, Query(min_length=2, max_length=500, description="Search text")],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    method: Literal["hybrid", "keyword", "vector"] = "hybrid",
    as_of: Annotated[date | None, Query(description="Only rules in force on this date")] = None,
    doc_type: Literal["circular", "master_direction", "notification"] | None = None,
    department: str | None = None,
) -> PassagesResponse:
    """The best-matching passages (chunks), with the offsets needed to highlight them."""
    filters = SearchFilters(as_of=as_of, doc_type=doc_type, department=department)
    hits = search_chunks(session, q, embedder, k=k, filters=filters, method=method)
    urls = {h.doc_id: detail_url(int(h.source_id)) for h in hits if h.source_id}
    return PassagesResponse(
        query=q,
        method=method,
        count=len(hits),
        results=[
            Passage(
                chunk_id=h.chunk_id,
                source_id=h.source_id,
                title=h.title,
                heading=h.heading,
                text=h.text,
                char_start=h.char_start,
                char_end=h.char_end,
                url=urls.get(h.doc_id, ""),
                score=h.score,
                keyword_rank=h.keyword_rank,
                vector_rank=h.vector_rank,
            )
            for h in hits
        ],
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    as_of: date | None = None  # answer with the rules in force on this date


class CitationOut(BaseModel):
    passage: str
    source_id: str | None
    quote: str
    verified: bool
    doc_char_start: int | None
    doc_char_end: int | None


class SentenceOut(BaseModel):
    text: str
    citations: list[CitationOut]


class PassageOut(BaseModel):
    label: str
    source_id: str | None
    title: str
    heading: str | None
    url: str
    issued_date: str | None
    in_force: bool | None


class AskResponse(BaseModel):
    question: str
    as_of: date | None
    abstained: bool
    answer: str
    note: str | None
    sentences: list[SentenceOut]
    passages: list[PassageOut]
    unsupported: list[str]


def llm_dependency() -> LLM:
    return LiteLLM()


@app.post("/ask")
def ask(
    req: AskRequest,
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(embedder_dependency)],
    llm: Annotated[LLM, Depends(llm_dependency)],
) -> AskResponse:
    """Answer from retrieved passages; every sentence carries verified quotes."""
    a = answer_question(session, req.question, llm, embedder, as_of=req.as_of)
    return AskResponse(
        question=a.question,
        as_of=a.as_of,
        abstained=a.abstained,
        answer=a.text,
        note=a.note,
        sentences=[
            SentenceOut(
                text=s.text,
                citations=[
                    CitationOut(
                        passage=c.label,
                        source_id=c.source_id,
                        quote=c.quote,
                        verified=c.verified,
                        doc_char_start=c.doc_char_start,
                        doc_char_end=c.doc_char_end,
                    )
                    for c in s.citations
                ],
            )
            for s in a.sentences
        ],
        passages=[
            PassageOut(
                label=p.label,
                source_id=p.source_id,
                title=p.title,
                heading=p.heading,
                url=p.url,
                issued_date=p.issued_date,
                in_force=p.in_force,
            )
            for p in a.passages
        ],
        unsupported=a.unsupported,
    )


class DocumentOut(BaseModel):
    source_id: str | None
    title: str
    circular_no: str | None
    rbi_no: str | None
    doc_type: str
    department: str | None
    issued_date: date
    updated_on: date | None
    text_as_of: date | None
    is_withdrawn: bool
    withdrawn_on: date | None
    valid_from: date | None
    valid_to: date | None
    url: str
    pdf_url: str | None
    raw_text: str | None


@app.get("/documents/{source_id}")
def get_document(source_id: str, session: Annotated[Session, Depends(get_session)]) -> DocumentOut:
    """A stored document with its full text, for showing a citation in context."""
    doc = session.scalar(select(Document).where(Document.source_id == source_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"no document {source_id}")
    return DocumentOut.model_validate(doc, from_attributes=True)
