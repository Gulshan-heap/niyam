# Niyam — which rule is in force on a given date?

Agentic RAG over Indian financial regulations (RBI first, SEBI next). Regulators keep issuing
circulars that amend or supersede older ones, so a plain "chat with PDFs" bot happily cites
rules that no longer apply. Niyam tracks those amendment links, answers *"what was the rule
as of date X"*, cites the exact highlighted passage, and alerts you on Telegram when a rule you
follow changes.

Inspired by [jamwithai/production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course).

## Status

| Week | Milestone | Status |
|---|---|---|
| 1 | Infra: Postgres + pgvector, FastAPI, Alembic, tests | ✅ |
| 2 | RBI ingestion (scraper, PDF parsing with page/bbox, daily job) | ⏳ |
| 3 | Keyword search + eval seed set | |
| 4 | Chunking + hybrid search (BM25 + vectors, RRF) | |
| 5 | RAG with sentence-level citations + Streamlit UI | |
| 6 | Temporal layer: amendment graph, as-of-date retrieval | |
| 7 | LangGraph agent + Telegram bot + change alerts | |
| 8 | Langfuse, cache, RAGAS dashboard, MCP server, deploy | |

## Quick start

Requires Docker Desktop and [uv](https://docs.astral.sh/uv/).

```bash
# Everything in containers
docker compose up --build
curl http://localhost:8000/health

# Or: DB in Docker, API locally (faster iteration)
cp .env.example .env
docker compose up -d db
uv sync
uv run alembic upgrade head
uv run uvicorn niyam.api.main:app --reload
```

## Tests

```bash
uv run pytest          # DB integration tests skip if Postgres isn't running
uv run ruff check .
```

## Schema

- `documents`: one row per circular / Master Direction, with `valid_from` / `valid_to`.
- `relations`: amends / supersedes / repeals edges between documents.
- `chunks`: text with page and bounding boxes (for highlighting), a generated `tsvector`
  (keyword search) and a 384-d `embedding` (vector search). Chunks copy their document's validity
  window, so "in force on date D" is a single `WHERE` clause.
- `subscriptions`, `alerts_sent`: Telegram follows and alert history.
- `query_cache`, `eval_runs`, `eval_results`: caching and the evaluation dashboard.
