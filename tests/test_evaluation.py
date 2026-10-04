import json
from datetime import date

import pytest

from niyam.evaluation.golden import GoldenQuestion, load_golden
from niyam.evaluation.metrics import (
    hit_at_k,
    question_metrics,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from niyam.evaluation.run import QuestionResult, RunConfig, RunReport


def test_recall_hit_and_reciprocal_rank():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 1) == 0.0
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0
    assert reciprocal_rank(ranked, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_recall_needs_relevant_documents():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_question_metrics_and_summary():
    q1 = question_metrics(["a", "b"], {"a"}, ks=(1, 5))
    q2 = question_metrics(["x", "a"], {"a"}, ks=(1, 5))
    assert q1 == {"hit@1": 1.0, "hit@5": 1.0, "recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0}
    assert summarize([q1, q2])["mrr"] == pytest.approx(0.75)
    assert summarize([q1, q2])["hit@1"] == 0.5
    assert summarize([]) == {}


def write(tmp_path, rows) -> object:
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    return p


def test_load_golden(tmp_path):
    p = write(
        tmp_path,
        [
            {"id": "q1", "question": "KYC?", "relevant": [12943], "tags": ["kyc"]},
            {"id": "q2", "question": "PSL in 2022?", "relevant": ["11959"], "as_of": "2022-06-01"},
        ],
    )
    q1, q2 = load_golden(p)
    assert q1.relevant == ["12943"]  # ids normalised to strings
    assert q1.as_of is None and q1.tags == ["kyc"]
    assert q2.as_of == date(2022, 6, 1)


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "q1", "question": "x", "relevant": []}],
        [{"id": "q1", "question": "x"}],
        [{"id": "q1", "question": "x", "relevant": ["1"], "as_of": "June 2022"}],
        [
            {"id": "q1", "question": "x", "relevant": ["1"]},
            {"id": "q1", "question": "y", "relevant": ["2"]},
        ],
    ],
)
def test_load_golden_rejects_bad_rows(tmp_path, rows):
    with pytest.raises(ValueError):
        load_golden(write(tmp_path, rows))


def test_report_groups_by_tag():
    def result(qid, tags, hit):
        q = GoldenQuestion(qid, "?", ["a"], tags=tags)
        return QuestionResult(
            q, ["a"] if hit else ["b"], question_metrics(["a"] if hit else ["b"], {"a"})
        )

    report = RunReport(
        RunConfig(),
        [result("1", ["kyc"], True), result("2", ["kyc", "nbfc"], False), result("3", [], True)],
    )
    assert report.summary["hit@1"] == pytest.approx(2 / 3)
    assert report.by_tag()["kyc"]["hit@1"] == 0.5
    assert report.by_tag()["nbfc"]["hit@1"] == 0.0
