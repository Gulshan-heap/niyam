from datetime import date

from niyam.ingest.__main__ import id_entries, month_range, previous_month, select_entries
from niyam.ingest.scrapers.rbi import ListingEntry


def test_month_range_crosses_year():
    assert list(month_range("2024-11", "2025-02")) == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    assert list(month_range("2024-03", "2024-03")) == [(2024, 3)]


def test_previous_month():
    assert previous_month(date(2026, 1, 15)) == (2025, 12)
    assert previous_month(date(2026, 10, 4)) == (2026, 9)


def test_select_entries_by_title_and_section():
    def e(title, section=None):
        return ListingEntry(1, title, None, None, section)

    entries = [
        e("Master Direction - KYC", "Commercial Banks"),
        e("Digital Lending Directions", "Non-Banking Financial Companies"),
        e("Currency chest operations"),
    ]
    assert [x.title for x in select_entries(entries, match="kyc|digital lending")] == [
        "Master Direction - KYC",
        "Digital Lending Directions",
    ]
    assert len(select_entries(entries, section="non-banking")) == 1


def test_id_entries_newest_first_inclusive():
    assert [e.rbi_id for e in id_entries(10, 13)] == [13, 12, 11, 10]
