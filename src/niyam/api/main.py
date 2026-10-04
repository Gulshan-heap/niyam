from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from niyam import __version__
from niyam.db.session import get_session
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
