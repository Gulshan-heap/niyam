from datetime import date

from niyam.evaluation.answers import score_answer, summarize_answers
from niyam.evaluation.golden import GoldenQuestion
from niyam.rag.answer import Answer
from niyam.rag.verify import Citation, Sentence

Q = GoldenQuestion("q1", "?", ["100"])
VALIDITY = {
    "100": (date(2020, 1, 1), None, False),
    "200": (date(2020, 1, 1), date(2021, 1, 1), True),
}


def state(answer):
    return {"answer": answer, "trace": []}


def test_scores_for_a_grounded_answer_citing_the_right_document():
    a = Answer(
        "?",
        date(2024, 1, 1),
        False,
        [Sentence("x", [Citation("P1", "q", True, 100, 1, "100")]),
         Sentence("y", [Citation("P2", "q", True, 100, 2, "200")])],
        [],
        unsupported=["dropped"],
    )  # fmt: skip
    s = score_answer(state(a), Q, VALIDITY)
    assert s["answered"] == 1.0
    assert s["grounded"] == 2 / 3
    assert s["cites_relevant"] == 1.0
    assert s["cites_in_force"] == 0.5  # "200" was withdrawn in 2021


def test_abstention_and_summary():
    abstained = score_answer(state(Answer("?", date(2024, 1, 1), True, [], [])), Q, VALIDITY)
    assert abstained == {"answered": 0.0}
    full = {"answered": 1.0, "grounded": 1.0, "cites_relevant": 0.0, "cites_in_force": 1.0}
    summary = summarize_answers([abstained, full])
    assert summary == {
        "answered": 0.5,
        "grounded": 1.0,
        "cites_relevant": 0.0,
        "cites_in_force": 1.0,
    }
