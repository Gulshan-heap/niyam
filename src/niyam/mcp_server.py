"""MCP server exposing Niyam to Claude Desktop, Cursor and other MCP clients.

    uv run python -m niyam.mcp_server        # stdio transport

Claude Desktop config (claude_desktop_config.json):

    {"mcpServers": {"niyam": {"command": "uv",
        "args": ["--directory", "D:/projects/niyam", "run", "python", "-m", "niyam.mcp_server"]}}}
"""

from datetime import date

from mcp.server.mcpserver import MCPServer
from sqlalchemy import select
from sqlalchemy.orm import Session

from niyam.db.models import Document, Relation
from niyam.db.session import get_engine
from niyam.retrieval.embeddings import get_embedder
from niyam.retrieval.hybrid import search_chunks
from niyam.retrieval.keyword import SearchFilters

server = MCPServer(
    "niyam",
    instructions=(
        "Search Reserve Bank of India (RBI) circulars and Master Directions, knowing which "
        "rule was in force on a given date. Pass as_of (YYYY-MM-DD) for past dates; results "
        "default to the rules in force today. Cite the returned URLs."
    ),
)


def _as_of(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


@server.tool()
def search_regulations(query: str, as_of: str | None = None, k: int = 5) -> list[dict]:
    """Passages from RBI documents in force on `as_of` (YYYY-MM-DD, default today) that best
    match `query`, with the document title, heading, issue date and URL."""
    with Session(get_engine()) as session:
        hits = search_chunks(
            session,
            query,
            get_embedder(),
            k=max(1, min(k, 20)),
            filters=SearchFilters(as_of=_as_of(as_of)),
        )
        docs = {
            d.id: d
            for d in session.scalars(
                select(Document).where(Document.id.in_([h.doc_id for h in hits]))
            )
        }
        return [
            {
                "source_id": h.source_id,
                "title": h.title,
                "heading": h.heading,
                "issued_date": docs[h.doc_id].issued_date.isoformat(),
                "url": docs[h.doc_id].url,
                "text": h.text,
            }
            for h in hits
        ]


@server.tool()
def get_circular(source_id: str) -> dict:
    """One RBI document by its page id (e.g. "13156"): title, numbers, dates, validity, URL
    and the first part of its text."""
    with Session(get_engine()) as session:
        d = session.scalar(select(Document).where(Document.source_id == source_id))
        if d is None:
            return {"error": f"no RBI document {source_id}"}
        return {
            "source_id": d.source_id,
            "title": d.title,
            "reference": d.circular_no,
            "rbi_no": d.rbi_no,
            "type": d.doc_type,
            "department": d.department,
            "issued_date": d.issued_date.isoformat(),
            "updated_on": d.updated_on.isoformat() if d.updated_on else None,
            "withdrawn": d.is_withdrawn,
            "valid_from": d.valid_from.isoformat() if d.valid_from else None,
            "valid_to": d.valid_to.isoformat() if d.valid_to else None,
            "url": d.url,
            "text_start": (d.raw_text or "")[:3000],
        }


@server.tool()
def get_amendment_chain(source_id: str) -> dict:
    """The documents that amended, superseded or repealed this one (newest first), and the
    ones it acts on."""
    with Session(get_engine()) as session:
        d = session.scalar(select(Document).where(Document.source_id == source_id))
        if d is None:
            return {"error": f"no RBI document {source_id}"}

        def rows(incoming: bool) -> list[dict]:
            other = Relation.src_doc_id if incoming else Relation.dst_doc_id
            this = Relation.dst_doc_id if incoming else Relation.src_doc_id
            q = (
                select(Document, Relation.type)
                .join(Relation, other == Document.id)
                .where(this == d.id, Relation.type != "refers")
                .order_by(Document.issued_date.desc())
            )
            return [
                {
                    "relation": t,
                    "source_id": o.source_id,
                    "title": o.title,
                    "issued_date": o.issued_date.isoformat(),
                    "url": o.url,
                }
                for o, t in session.execute(q)
            ]

        return {
            "source_id": d.source_id,
            "title": d.title,
            "changed_by": rows(True),
            "changes": rows(False),
        }


@server.tool()
def ask(question: str, as_of: str | None = None) -> dict:
    """A cited answer from RBI documents (needs an LLM key). Each sentence comes with the
    verified quotes and source URLs; abstains when the sources don't answer."""
    from niyam.agent.graph import run_agent
    from niyam.rag.llm import LiteLLM, grader_llm

    with Session(get_engine()) as session:
        state = run_agent(
            session,
            question,
            LiteLLM(),
            get_embedder(),
            as_of=date.fromisoformat(as_of) if as_of else None,
            grader=grader_llm(),
        )
    a = state["answer"]
    urls = {p.label: p.url for p in a.passages}
    return {
        "as_of": a.as_of.isoformat(),
        "abstained": a.abstained,
        "note": a.note,
        "sentences": [
            {
                "text": s.text,
                "sources": [
                    {"url": urls.get(c.label), "quote": c.quote} for c in s.citations if c.verified
                ],
            }
            for s in a.sentences
        ],
    }


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
