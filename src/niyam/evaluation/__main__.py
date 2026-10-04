"""Evaluate retrieval against the golden set.

python -m niyam.evaluation                        # keyword search, saved to eval_runs
python -m niyam.evaluation --mode all --no-save   # try a variant without recording it
python -m niyam.evaluation --misses               # list questions that missed
"""

import argparse
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from niyam.config import get_settings
from niyam.db.session import get_engine
from niyam.evaluation.golden import load_golden
from niyam.evaluation.run import RETRIEVERS, RunConfig, RunReport, run_eval, save_run
from niyam.retrieval.embeddings import get_embedder

DEFAULT_GOLDEN = Path("eval/golden.jsonl")
COLUMNS = ["hit@1", "hit@5", "hit@10", "recall@10", "mrr", "in_force@5"]


def format_report(report: RunReport, show_misses: bool = False) -> str:
    lines = [f"{'':14}{'n':>4}" + "".join(f"{c:>11}" for c in COLUMNS)]

    def row(label: str, n: int, scores: dict[str, float]) -> str:
        return f"{label:14}{n:>4}" + "".join(f"{scores[c]:>11.3f}" for c in COLUMNS)

    lines.append(row("all", len(report.results), report.summary))
    for tag, scores in report.by_tag().items():
        n = sum(tag in r.question.tags for r in report.results)
        lines.append(row(f"  {tag}", n, scores))
    if show_misses:
        misses = [r for r in report.results if r.scores["hit@10"] == 0]
        lines.append(f"\nmissed in top {report.config.k}: {len(misses)}")
        for r in misses:
            lines.append(f"  {r.question.id}: {r.question.question}")
            lines.append(f"      want {r.question.relevant}, got {r.ranked[:5]}")
    if report.unknown_relevant:
        lines.append(f"\nrelevant ids missing from the database: {report.unknown_relevant}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m niyam.evaluation")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--retriever", choices=RETRIEVERS, default="keyword-doc")
    parser.add_argument("--mode", choices=["any", "all"], default="any")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--norm", type=int, default=RunConfig.normalization, help="ts_rank_cd normalization"
    )
    parser.add_argument("--no-as-of", action="store_true", help="ignore questions' as_of dates")
    parser.add_argument("--no-save", action="store_true", help="don't record in eval_runs")
    parser.add_argument("--misses", action="store_true", help="list questions that missed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    config = RunConfig(
        retriever=args.retriever,
        mode=args.mode,
        k=args.k,
        normalization=args.norm,
        apply_as_of=not args.no_as_of,
    )
    embedder = get_embedder() if args.retriever in ("vector", "hybrid") else None
    questions = load_golden(args.golden)
    with Session(get_engine()) as session:
        report = run_eval(session, questions, config, embedder)
        print(format_report(report, show_misses=args.misses))
        if not args.no_save:
            print(f"\nsaved as eval run {save_run(session, report, args.golden)}")


if __name__ == "__main__":
    main()
