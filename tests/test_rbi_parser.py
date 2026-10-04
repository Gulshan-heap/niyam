"""Parser tests against trimmed copies of real rbi.org.in pages (tests/fixtures/rbi)."""

from datetime import date
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from niyam.ingest.scrapers.rbi import (
    _department,
    detail_url,
    html_to_blocks,
    is_master_direction,
    parse_date,
    parse_detail,
    parse_form_state,
    parse_listing,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def blocks_of(html: str) -> list[str]:
    return html_to_blocks(BeautifulSoup(html, "lxml").body)


# ---------- listings ----------


def test_notification_listing():
    entries = parse_listing(fixture("listing_2024_03.html"))
    assert len(entries) == 11
    first = entries[0]
    assert first.rbi_id == 12641
    assert first.title == "Currency Chests (CCs) operations on March 31, 2024"
    assert first.listed_date == date(2024, 3, 27)
    assert first.pdf_url.startswith("https://rbidocs.rbi.org.in/")
    assert entries[-1].listed_date == date(2024, 3, 5)
    assert all(e.section is None for e in entries)


def test_form_state_has_postback_fields():
    state = parse_form_state(fixture("listing_2024_03.html"))
    assert state["__VIEWSTATE"] == "VIEWSTATE-FIXTURE"
    assert {"__EVENTVALIDATION", "hdnYear", "hdnMonth", "UsrFontCntr$btn"} <= state.keys()


def test_master_direction_index_tracks_sections_and_dates():
    entries = parse_listing(fixture("md_index.html"))
    first = entries[0]
    assert first.rbi_id == 11322
    assert first.section == "Banker and Debt Manager to Government"
    assert first.listed_date == date(2018, 7, 3)
    # "(Updated as on ...)" is stripped from titles
    pd = next(e for e in entries if e.rbi_id == 10476)
    assert pd.title == "Master Direction – Operational Guidelines for Primary Dealers"
    assert pd.updated_on == date(2018, 11, 22)
    assert first.updated_on is None
    assert {e.section for e in entries} >= {"Banker to Governments and Banks"}


# ---------- detail pages ----------


def test_detail_recent_amendment():
    d = parse_detail(fixture("detail_13722.html"), 13722)
    assert d.title.startswith("Reserve Bank of India (Local Area Banks")
    assert d.rbi_no == "RBI/2026-27/278"
    assert d.ref_no == "DOR.HOL.REC.No.238/16.13.100/2026-27"
    assert d.department == "Department of Regulation"
    assert d.issued_date == date(2026, 10, 1)
    assert d.updated_on is None
    assert d.withdrawn_on is None
    assert d.linked_ids == [13086]  # the Master Direction it amends
    assert "4. These Directions shall come into force with immediate effect." in d.blocks
    assert not is_master_direction(d.title)


def test_detail_withdrawn_master_direction():
    d = parse_detail(fixture("detail_11566_md_withdrawn.html"), 11566)
    assert d.title == "Master Direction - Know Your Customer (KYC) Direction, 2016"
    assert d.rbi_no == "RBI/DBR/2015-16/18"
    assert d.ref_no == "DBR.AML.BC.No.81/14.01.001/2015-16"
    assert d.department == "Department of Banking Regulation"
    assert d.issued_date == date(2016, 2, 25)
    assert d.updated_on == date(2025, 8, 14)  # latest of many "Updated as on" stamps
    assert d.withdrawn_on == date(2025, 12, 4)  # from the Withdrawn04122025.jpg watermark
    assert is_master_direction(d.title)
    # inline footnote markers are dropped, the text around them is kept
    para = next(b for b in d.blocks if b.startswith("(a) The provisions of these Directions"))
    assert "(a) 3" not in para


def test_detail_old_circular():
    d = parse_detail(fixture("detail_9914_old.html"), 9914)
    assert d.rbi_no == "RBI/2015-16/108"
    assert d.ref_no == "DNBR (PD) CC No. 051/03.10.119/2015-16"
    assert d.department == "Department of Non-Banking Regulation"
    assert d.issued_date == date(2015, 7, 1)
    assert d.updated_on is None  # "updated as on June 30, 2015" in prose is not a stamp
    assert d.withdrawn_on == date(2025, 12, 4)
    assert d.blocks[2].split("\n") == [
        "To",
        "All Non-Banking Financial Companies (NBFCs),",
        "Miscellaneous Non-Banking Companies (MNBCs),",
        "and Residuary Non-Banking Companies (RNBCs)",
    ]


def test_detail_text_joins_blocks():
    d = parse_detail(fixture("detail_13722.html"), 13722)
    assert d.text.startswith(
        "RBI/2026-27/278\nDOR.HOL.REC.No.238/16.13.100/2026-27\n\nOctober 1, 2026"
    )


def test_detail_rejects_unexpected_page():
    with pytest.raises(ValueError):
        parse_detail("<html><body>Unauthorised Access</body></html>", 1)


# ---------- helpers ----------


def test_blocks_split_paragraphs_nested_in_spans():
    html = "<span>1 INTRODUCTION<p>First para.</p><p>Second   para.</p></span>"
    assert blocks_of(html) == ["1 INTRODUCTION", "First para.", "Second para."]


def test_blocks_tables_become_rows():
    html = (
        "<table><tr><td>Limit</td><td>₹ 50,000</td></tr>"
        "<tr><td>Tenor</td><td>1 yr</td></tr></table>"
    )
    assert blocks_of(html) == ["Limit | ₹ 50,000", "Tenor | 1 yr"]


def test_blocks_keep_line_breaks_and_drop_footnote_markers():
    html = '<p>RBI/2024-25/1<br>DOR.No.5</p><p>Rule<sup><a href="#F1">7</a></sup> text.</p>'
    assert blocks_of(html) == ["RBI/2024-25/1\nDOR.No.5", "Rule text."]


@pytest.mark.parametrize(
    "s,expected",
    [
        ("October 1, 2026", date(2026, 10, 1)),
        ("Mar 05, 2024", date(2024, 3, 5)),
        ("Sept. 3, 2019", date(2019, 9, 3)),
        ("no date here", None),
    ],
)
def test_parse_date(s, expected):
    assert parse_date(s) == expected


@pytest.mark.parametrize(
    "ref,dept",
    [
        ("CO.DPSS.POLC.No.S-123/02.14.003/2024-25", "Department of Payment and Settlement Systems"),
        ("A.P. (DIR Series) Circular No. 05", "Foreign Exchange Department"),
        ("DoR.FIN.REC.No.45/03.10.123/2023-24", "Department of Regulation"),
        ("Something else", None),
        (None, None),
    ],
)
def test_department_from_reference(ref, dept):
    assert _department(ref) == dept


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Master Direction - Know Your Customer (KYC) Direction, 2016", True),
        ("Reserve Bank of India (Local Area Banks - ...) Directions, 2025", True),
        ("Reserve Bank of India (Local Area Banks - ...) Amendment Directions, 2026", False),
        ("Master Circular – KYC Guidelines", False),
    ],
)
def test_is_master_direction(title, expected):
    assert is_master_direction(title) is expected


def test_detail_url_is_canonical():
    assert (
        detail_url(13086) == "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=13086&Mode=0"
    )


def test_detail_fema_notification():
    d = parse_detail(fixture("detail_13714_fema.html"), 13714)
    assert d.rbi_no is None
    assert d.ref_no == "Notification No. FEMA 23(R)/(1)/2026-RB"
    assert d.department == "Foreign Exchange Department"
    assert d.issued_date == date(2026, 9, 22)
    assert d.title.startswith("Foreign Exchange Management (Export and Import")


def test_department_from_letterhead():
    assert _department(None, "RESERVE BANK OF INDIA\nFOREIGN EXCHANGE DEPARTMENT") == (
        "Foreign Exchange Department"
    )
