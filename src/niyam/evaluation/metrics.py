"""Retrieval metrics over ranked document ids."""

from collections.abc import Iterable, Sequence
from statistics import mean


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        raise ValueError("a question needs at least one relevant document")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top k."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document (0 if none was retrieved)."""
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def question_metrics(
    ranked: Sequence[str], relevant: set[str], ks: Iterable[int] = (1, 5, 10)
) -> dict[str, float]:
    scores = {f"hit@{k}": hit_at_k(ranked, relevant, k) for k in ks}
    scores |= {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks}
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    return scores


def summarize(per_question: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_question:
        return {}
    return {name: mean(q[name] for q in per_question) for name in per_question[0]}
