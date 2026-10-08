"""Answer-level evaluation: run the full agent on every golden question and score answers.

    python -m niyam.evaluation.answers            # all questions, saved to eval_runs
    python -m niyam.evaluation.answers --limit 5 --no-save

Per question (scores are means over answered questions unless noted):
- answered: 1 if the agent did not abstain (mean over all questions);
- grounded: verified sentences / all sentences the model wrote;
- cites_relevant: 1 if a verified citation points at one of the question's documents;
- cites_in_force: share of cited documents in force on the question's date.

Slow by design on free LLM tiers: --pause spaces questions to stay under rate limits.
"""

import argparse
import logging
import time
from datetime import date
from pathlib import Path
from statistics import mean

from sqlalchemy import select
from sqlalchemy.orm import Session

from niyam.agent.graph import run_agent
from niyam.config import get_settings
from niyam.db.models import Document, EvalResult, EvalRun
from niyam.db.session import get_engine
from niyam.evaluation.golden import GoldenQuestion, load_golden
from niyam.evaluation.run import git_dirty, git_sha, in_force_rate
from niyam.rag.llm import LLM, LiteLLM, grader_llm
from niyam.retrieval.embeddings import Embedder, get_embedder

log = logging.getLogger("niyam.evaluation.answers")


def score_answer(state: dict, q: GoldenQuestion, validity: dict) -> dict[str, float]:
    a = state["answer"]
    if a.abstained:
        return {"answered": 0.0}
    cited = [c.source_id for s in a.sentences for c in s.citations if c.verified and c.source_id]
    total = len(a.sentences) + len(a.unsupported)
    return {
        "answered": 1.0,
        "grounded": len(a.sentences) / total if total else 0.0,
        "cites_relevant": 1.0 if set(cited) & set(q.relevant) else 0.0,
        "cites_in_force": in_force_rate(list(dict.fromkeys(cited)), a.as_of, validity),
    }


def summarize_answers(per_question: list[dict[str, float]]) -> dict[str, float]:
    if not per_question:
        return {}
    answered = [p for p in per_question if p["answered"]]
    out = {"answered": mean(p["answered"] for p in per_question)}
    for key in ("grounded", "cites_relevant", "cites_in_force"):
        out[key] = mean(p[key] for p in answered) if answered else 0.0
    return out


def evaluate_answers(
    session: Session,
    questions: list[GoldenQuestion],
    llm: LLM,
    grader: LLM,
    embedder: Embedder,
    pause: float = 0.0,
) -> list[tuple[GoldenQuestion, dict, dict[str, float]]]:
    validity = {
        sid: (vf, vt, wd)
        for sid, vf, vt, wd in session.execute(
            select(
                Document.source_id, Document.valid_from, Document.valid_to, Document.is_withdrawn
            )
        )
    }
    results = []
    for i, q in enumerate(questions, start=1):
        try:
            state = run_agent(session, q.question, llm, embedder, as_of=q.as_of, grader=grader)
            scores = score_answer(state, q, validity)
        except Exception as exc:  # a failed call shouldn't sink the whole run
            log.warning("%s failed: %s", q.id, exc)
            state, scores = {"trace": [f"error: {exc}"]}, {"answered": 0.0}
        results.append((q, state, scores))
        log.info("%d/%d %s %s", i, len(questions), q.id, scores)
        if pause and i < len(questions):
            time.sleep(pause)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m niyam.evaluation.answers")
    parser.add_argument("--golden", type=Path, default=Path("eval/golden.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between questions")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(message)s")
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    questions = load_golden(args.golden)[: args.limit]
    with Session(get_engine()) as session:
        results = evaluate_answers(
            session, questions, LiteLLM(), grader_llm(), get_embedder(), args.pause
        )
        summary = summarize_answers([s for _, _, s in results])
        print("\n".join(f"{k:>16}: {v:.3f}" for k, v in summary.items()))
        for q, state, s in results:
            if not s.get("answered") or not s.get("cites_relevant"):
                a = state.get("answer")
                cited = sorted(
                    {
                        c.source_id
                        for x in getattr(a, "sentences", [])
                        for c in x.citations
                        if c.verified
                    }
                )
                print(
                    f"\n{q.id}: {q.question}\n  {s}\n  want {q.relevant[:5]}, cited {cited}"
                    f"  note={getattr(a, 'note', None)}"
                )
        if not args.no_save:
            run = EvalRun(
                git_sha=git_sha(),
                config_json={
                    "retriever": "answer-agent",
                    "apply_as_of": True,
                    "git_dirty": git_dirty(),
                    "llm_model": get_settings().llm_model,
                    "grader_model": get_settings().llm_grader_model,
                    "questions": len(results),
                    "summary": summary,
                    "date": date.today().isoformat(),
                },
            )
            session.add(run)
            session.flush()
            session.add_all(
                EvalResult(run_id=run.id, qid=q.id, metric=k, score=v)
                for q, _, s in results
                for k, v in s.items()
            )
            session.commit()
            print(f"\nsaved as eval run {run.id}")


if __name__ == "__main__":
    main()
