from datetime import date

import pytest

from niyam.rag.dates import parse_as_of

TODAY = date(2026, 10, 4)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What was the KYC rule as of 1 June 2024?", date(2024, 6, 1)),
        ("What was the rule on 1st June, 2024?", date(2024, 6, 1)),
        ("credit card closure rules as of June 1, 2024", date(2024, 6, 1)),
        ("PSL targets as on 2023-03-31", date(2023, 3, 31)),
        ("gold loan LTV as of 15.08.2025", date(2025, 8, 15)),
        ("What was the gold loan LTV in August 2025?", date(2025, 8, 31)),
        ("What was the overall PSL target in 2023?", date(2023, 12, 31)),
        ("digital lending rules before May 2025", date(2025, 4, 30)),
        ("What did the rules say back in 2019?", date(2019, 12, 31)),
        ("What changed in September 2026?", date(2026, 9, 30)),
        ("rules in force in 2026", TODAY),  # current period: capped at today
        ("PSL rule before 2024", date(2023, 12, 31)),
    ],
)
def test_dates_introduced_by_a_time_word(question, expected):
    assert parse_as_of(question, today=TODAY).on == expected


@pytest.mark.parametrize(
    "question",
    [
        "What does the KYC Direction, 2016 say about periodic updation?",
        "Powers under the Banking Regulation Act, 1949",
        "Explain FEMA 23(R)/2015-RB",
        "What is the current cooling-off period?",
        "the 2025-26 reference DOR.AML.REC.44/14.01.001/2023-24",
        "rules in 2030",  # future: not a date we can answer about
    ],
)
def test_no_date(question):
    assert parse_as_of(question, today=TODAY) is None


def test_reports_the_words_it_used():
    p = parse_as_of("What was the PSL target in March 2024 for banks?", today=TODAY)
    assert p.text == "in March 2024"
