from datetime import date, datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from niyam import __version__
from niyam.agent.graph import run_agent
from niyam.db.models import Document, EvalRun, Relation
from niyam.db.session import get_session
from niyam.ingest.scrapers.rbi import detail_url
from niyam.rag.cache import corpus_version, get_cached, put_cached
from niyam.rag.dates import resolve_as_of
from niyam.rag.llm import LLM, LiteLLM, grader_llm
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
    as_of_source: str  # request | question | today
    trace: list[str]  # the agent's steps, for transparency and debugging
    cached: bool = False


def llm_dependency() -> LLM:
    return LiteLLM()


def grader_dependency() -> LLM:
    return grader_llm()


@app.post("/ask")
def ask(
    req: AskRequest,
    session: Annotated[Session, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(embedder_dependency)],
    llm: Annotated[LLM, Depends(llm_dependency)],
    grader: Annotated[LLM, Depends(grader_dependency)],
) -> AskResponse:
    """Answer with the rules in force on the date asked (given, read from the question, or
    today). Every sentence carries quotes verified against the source passage. Answers are
    cached per question, date and corpus version."""
    as_of, _ = resolve_as_of(req.question, req.as_of)
    version = corpus_version(session)
    qvec = embedder.embed_query(req.question)
    if cached := get_cached(session, req.question, as_of, version, qvec):
        return AskResponse(**{**cached, "cached": True})

    state = run_agent(session, req.question, llm, embedder, as_of=req.as_of, grader=grader)
    a = state["answer"]
    response = AskResponse(
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
        as_of_source=state["as_of_source"],
        trace=state["trace"],
    )
    put_cached(session, req.question, as_of, version, response.model_dump(mode="json"), qvec)
    return response


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


class RelatedDoc(BaseModel):
    source_id: str | None
    title: str
    issued_date: date
    is_withdrawn: bool
    url: str
    method: str
    evidence: str | None


class TimelineOut(BaseModel):
    source_id: str | None
    title: str
    issued_date: date
    valid_from: date | None
    valid_to: date | None
    is_withdrawn: bool
    amended_by: list[RelatedDoc]  # newest first
    amends: list[RelatedDoc]
    superseded_by: list[RelatedDoc]
    supersedes: list[RelatedDoc]


@app.get("/documents/{source_id}/timeline")
def document_timeline(
    source_id: str, session: Annotated[Session, Depends(get_session)]
) -> TimelineOut:
    """How a document fits in the amendment chain: what changed it, and what it changed."""
    doc = session.scalar(select(Document).where(Document.source_id == source_id))
    if doc is None:
        raise HTTPException(status_code=404, detail=f"no document {source_id}")

    def related(direction: str, types: tuple[str, ...]) -> list[RelatedDoc]:
        other_id = Relation.src_doc_id if direction == "in" else Relation.dst_doc_id
        this_id = Relation.dst_doc_id if direction == "in" else Relation.src_doc_id
        rows = session.execute(
            select(Document, Relation)
            .join(Relation, other_id == Document.id)
            .where(this_id == doc.id, Relation.type.in_(types))
            .order_by(Document.issued_date.desc())
        ).all()
        return [
            RelatedDoc(
                source_id=d.source_id,
                title=d.title,
                issued_date=d.issued_date,
                is_withdrawn=d.is_withdrawn,
                url=d.url,
                method=r.method,
                evidence=r.evidence_text,
            )
            for d, r in rows
        ]

    return TimelineOut(
        source_id=doc.source_id,
        title=doc.title,
        issued_date=doc.issued_date,
        valid_from=doc.valid_from,
        valid_to=doc.valid_to,
        is_withdrawn=doc.is_withdrawn,
        amended_by=related("in", ("amends",)),
        amends=related("out", ("amends",)),
        superseded_by=related("in", ("supersedes", "repeals")),
        supersedes=related("out", ("supersedes", "repeals")),
    )


class EvalRunOut(BaseModel):
    id: int
    created_at: datetime
    git_sha: str | None
    git_dirty: bool | None
    retriever: str | None
    apply_as_of: bool | None
    questions: int | None
    summary: dict[str, float]


@app.get("/eval/runs")
def eval_runs(session: Annotated[Session, Depends(get_session)]) -> list[EvalRunOut]:
    """Saved evaluation runs, oldest first, with their summary metrics."""
    runs = session.scalars(select(EvalRun).order_by(EvalRun.id)).all()
    return [
        EvalRunOut(
            id=r.id,
            created_at=r.created_at,
            git_sha=r.git_sha,
            git_dirty=(r.config_json or {}).get("git_dirty"),
            retriever=(r.config_json or {}).get("retriever"),
            apply_as_of=(r.config_json or {}).get("apply_as_of"),
            questions=(r.config_json or {}).get("questions"),
            summary=(r.config_json or {}).get("summary") or {},
        )
        for r in runs
    ]
