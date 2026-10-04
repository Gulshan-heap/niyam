"""Read the date a question is about ("what was the KYC rule in 2023?").

Only dates introduced by a time word count, so titles such as "KYC Direction, 2016" or
"Banking Regulation Act, 1949" are not mistaken for the date asked about. Vague dates are
read near their middle, away from period edges: a year as July 1, a month as the 15th.
"before X" means the day before X.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MONTH = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_CUE = r"(?P<cue>as of|as on|as at|in|during|on|before|until|till|by|back in|effective)"
_YEAR = r"(?P<year>(?:19|20)\d{2})(?!\d|-\d)"

PATTERNS = [
    # on 1 June 2024 / as of 1st June, 2024
    re.compile(
        rf"\b{_CUE}\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH}),?\s+{_YEAR}", re.I
    ),
    # as of June 1, 2024
    re.compile(
        rf"\b{_CUE}\s+(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR}", re.I
    ),
    # as of 2024-06-01 / 01.06.2024 / 01/06/2024
    re.compile(rf"\b{_CUE}\s+(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})", re.I),
    re.compile(rf"\b{_CUE}\s+(?P<dmy>\d{{1,2}}[./]\d{{1,2}}[./](?:19|20)\d{{2}})", re.I),
    # in March 2024
    re.compile(rf"\b{_CUE}\s+(?P<month>{_MONTH}),?\s+{_YEAR}", re.I),
    # in 2023
    re.compile(rf"\b{_CUE}\s+(?:the\s+year\s+)?{_YEAR}", re.I),
]


@dataclass
class ParsedDate:
    on: date
    text: str  # the words it was read from


def _build(m: re.Match) -> date | None:
    g = m.groupdict()
    try:
        if g.get("iso"):
            return date.fromisoformat(g["iso"])
        if g.get("dmy"):
            d, mo, y = (int(x) for x in re.split(r"[./]", g["dmy"]))
            return date(y, mo, d)
        year = int(g["year"])
        if g.get("month"):
            month = _MONTHS[g["month"].lower().rstrip(".")]
            return date(year, month, int(g["day"]) if g.get("day") else 15)
        return date(year, 7, 1)
    except (ValueError, KeyError):
        return None


def parse_as_of(question: str, today: date | None = None) -> ParsedDate | None:
    """The date the question asks about, or None when it names none (or a future one)."""
    today = today or date.today()
    for pattern in PATTERNS:
        for m in pattern.finditer(question):
            on = _build(m)
            if on is None:
                continue
            if m.group("cue").lower() in ("before", "until", "till", "by"):
                on -= timedelta(days=1)
            if on <= today:
                return ParsedDate(on, m.group(0))
    return None


def resolve_as_of(question: str, given: date | None = None) -> tuple[date, str]:
    """The date to answer for and where it came from: the request, the question, or today."""
    if given:
        return given, "request"
    if parsed := parse_as_of(question):
        return parsed.on, "question"
    return date.today(), "today"
