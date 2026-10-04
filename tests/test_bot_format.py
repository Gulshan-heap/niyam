from datetime import date

from niyam.bot.send import format_answer
from niyam.rag.answer import Answer
from niyam.rag.verify import Citation, PassageRef, Sentence


def passage(label, sid, title):
    return PassageRef(label, 1, sid, title, None, "...", 0, f"https://example.test/{sid}")


def test_answer_with_numbered_sources_and_quotes():
    p1 = passage("P1", "13156", "Credit Facilities <Directions>")
    p2 = passage("P2", "13559", "Fifth Amendment")
    a = Answer(
        "q",
        date(2026, 10, 4),
        False,
        [
            Sentence("At least one day.", [Citation("P2", "not less than one day", True, 100)]),
            Sentence("Set by the bank.", [Citation("P1", "determined by the bank", True, 100),
                                          Citation("P2", "invented", False, 40)]),
        ],
        [p1, p2],
    )  # fmt: skip
    text = format_answer(a, "today")
    assert text.startswith("<i>Rules in force on 04 Oct 2026 (today)</i>")
    assert "At least one day. [1]" in text and "Set by the bank. [2]" in text
    assert '[1] <a href="https://example.test/13559">Fifth Amendment</a>' in text
    assert "Credit Facilities &lt;Directions&gt;" in text  # HTML-escaped
    assert "invented" not in text  # unverified quotes are not shown


def test_abstention_note():
    a = Answer("q", date(2024, 3, 15), True, [], [], note="Not covered.")
    assert format_answer(a, "question") == (
        "<i>Rules in force on 15 Mar 2024 (from your question)</i>\n\nNot covered."
    )
