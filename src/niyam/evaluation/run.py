"""Run the golden set through the retriever, score it, and record the run in eval_runs."""

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from niyam.db.models import Document, EvalResult, EvalRun
from niyam.evaluation.golden import GoldenQuestion
from niyam.evaluation.metrics import question_metrics, summarize
from niyam.retrieval.embeddings import Embedder
from niyam.retrieval.hybrid import rank_documents, search_chunks
from niyam.retrieval.keyword import RANK_NORMALIZATION, Mode, SearchFilters, keyword_search

log = logging.getLogger(__name__)


RETRIEVERS = ("keyword-doc", "keyword-chunk", "vector", "hybrid")
CHUNKS_PER_QUESTION = 50  # chunk hits rolled up into the top-k documents


@dataclass
class RunConfig:
    retriever: str = "keyword-doc"  # one of RETRIEVERS
    mode: Mode = "any"
    k: int = 10
    normalization: int = RANK_NORMALIZATION  # ts_rank_cd length normalization flags
    apply_as_of: bool = True  # filter to documents in force on a question's as_of date


@dataclass
class QuestionResult:
    question: GoldenQuestion
    ranked: list[str]
    scores: dict[str, float]


@dataclass
class RunReport:
    config: RunConfig
    results: list[QuestionResult]
    unknown_relevant: dict[str, list[str]] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, float]:
        return summarize([r.scores for r in self.results])

    def by_tag(self) -> dict[str, dict[str, float]]:
        tags = sorted({t for r in self.results for t in r.question.tags})
        return {
            t: summarize([r.scores for r in self.results if t in r.question.tags]) for t in tags
        }


def retrieve(
    session: Session, q: GoldenQuestion, config: RunConfig, embedder: Embedder | None
) -> list[str]:
    """Ranked document ids (source_id) for one question."""
    filters = SearchFilters(as_of=q.as_of if config.apply_as_of else None)
    if config.retriever == "keyword-doc":
        hits = keyword_search(
            session,
            q.question,
            k=config.k,
            filters=filters,
            mode=config.mode,
            normalization=config.normalization,
        )
        return [h.source_id for h in hits if h.source_id]
    method = {"keyword-chunk": "keyword", "vector": "vector", "hybrid": "hybrid"}[config.retriever]
    hits = search_chunks(
        session, q.question, embedder, k=CHUNKS_PER_QUESTION, filters=filters, method=method
    )
    return rank_documents(hits)[: config.k]


def run_eval(
    session: Session,
    questions: list[GoldenQuestion],
    config: RunConfig,
    embedder: Embedder | None = None,
) -> RunReport:
    known = set(session.scalars(select(Document.source_id)).all())
    unknown = {q.id: [r for r in q.relevant if r not in known] for q in questions}
    unknown = {qid: ids for qid, ids in unknown.items() if ids}
    for qid, ids in unknown.items():
        log.warning("%s: relevant documents not in the database: %s", qid, ids)

    results = []
    for q in questions:
        ranked = retrieve(session, q, config, embedder)
        results.append(QuestionResult(q, ranked, question_metrics(ranked, set(q.relevant))))
    return RunReport(config, results, unknown)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def save_run(session: Session, report: RunReport, golden_path: Path) -> int:
    run = EvalRun(
        git_sha=git_sha(),
        config_json={
            **vars(report.config),
            "golden_file": golden_path.as_posix(),
            "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
            "questions": len(report.results),
            "summary": report.summary,
        },
    )
    session.add(run)
    session.flush()
    session.add_all(
        EvalResult(run_id=run.id, qid=r.question.id, metric=name, score=score)
        for r in report.results
        for name, score in r.scores.items()
    )
    session.commit()
    return run.id
