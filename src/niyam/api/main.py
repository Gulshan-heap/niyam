from typing import Annotated

from fastapi import Depends, FastAPI, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from niyam import __version__
from niyam.db.session import get_session

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
