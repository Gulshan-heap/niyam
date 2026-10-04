"""Ingestion pipeline against the real Postgres (skipped when unreachable), using fixture pages.

The pipeline commits after every document; the session joins the test transaction via
savepoints, so everything is rolled back at the end.
"""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from niyam.db.models import EMBEDDING_DIM, Chunk, Document
from niyam.db.session import get_engine
from niyam.ingest.http import BlockedError, FetchError
from niyam.ingest.pipeline import RbiIngestor
from niyam.ingest.scrapers.rbi import ListingEntry, Version

FIXTURES = Path(__file__).parent / "fixtures" / "rbi"
# Test ids far above real RBI ids so a dev database with real data never collides.
AMENDMENT, KYC_MD, OLD_MC, FEMA, PDF_ONLY = 9_013_722, 9_011_566, 9_009_914, 9_013_714, 9_013_136
PAGES = {
    PDF_ONLY: "detail_13136_pdf_only.html",
    AMENDMENT: "detail_13722.html",
    FEMA: "detail_13714_fema.html",
    KYC_MD: "detail_11566_md_withdrawn.html",
    OLD_MC: "detail_9914_old.html",
}


@pytest.fixture
def session():
    try:
        conn = get_engine().connect()
    except OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d db`")
    trans = conn.begin()
    with Session(bind=conn, join_transaction_mode="create_savepoint") as s:
        yield s
    trans.rollback()
    conn.close()


def _unique(rbi_id: int, html: str) -> str:
    """Add a marker paragraph so fixture text never equals the real page stored in a dev
    database, which the pipeline would report as a duplicate."""
    row = '<tr class="tablecontent2"><td>'
    return html.replace(row, f"{row}<p>[test fixture {rbi_id}]</p>", 1)


class FakeScraper:
    """Serves fixture HTML and writes it to the cache dir like PoliteClient would."""

    def __init__(self, cache_dir: Path):
        self.client = SimpleNamespace(cache_dir=cache_dir)
        self.pages = {
            i: _unique(i, (FIXTURES / name).read_text(encoding="utf-8"))
            for i, name in PAGES.items()
        }
        # rbi_id -> [(fixture, hist_id, as_of)], newest first
        self.versions = {PDF_ONLY: [("version_13136_h336.html", 336, date(2025, 11, 28))]}
        self.fetched: list[tuple[int, bool]] = []
        self.errors: dict[int, Exception] = {}

    def fetch_detail(self, rbi_id: int, refresh: bool = False):
        self.fetched.append((rbi_id, refresh))
        if rbi_id in self.errors:
            raise self.errors[rbi_id]
        path = self.client.cache_dir / f"rbi/notifications/{rbi_id}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.pages[rbi_id], encoding="utf-8")
        return self.pages[rbi_id], path

    def list_versions(self, rbi_id: int) -> list[Version]:
        return [
            Version(hist_id=h, as_of=as_of, url=f"v/{rbi_id}/{h}")
            for _, h, as_of in self.versions.get(rbi_id, [])
        ]

    def fetch_version(self, rbi_id: int, version: Version):
        name = next(n for n, h, _ in self.versions[rbi_id] if h == version.hist_id)
        html = _unique(rbi_id, (FIXTURES / name).read_text(encoding="utf-8"))
        path = self.client.cache_dir / f"rbi/notifications/{rbi_id}-v{version.hist_id}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return html, path


@pytest.fixture
def scraper(tmp_path):
    return FakeScraper(tmp_path / "raw")


@pytest.fixture
def ingestor(session, scraper, tmp_path):
    return RbiIngestor(session, scraper, data_dir=tmp_path)


def entry(rbi_id: int, updated_on: date | None = None) -> ListingEntry:
    return ListingEntry(
        rbi_id=rbi_id,
        title="listing title",
        listed_date=date(2026, 1, 1),
        pdf_url="https://rbidocs.rbi.org.in/x.PDF",
        updated_on=updated_on,
    )


def get_doc(session: Session, rbi_id: int) -> Document:
    return session.scalars(select(Document).where(Document.source_id == str(rbi_id))).one()


def test_new_document_fields(ingestor, session):
    stats = ingestor.ingest([entry(AMENDMENT)])
    assert stats.new == 1
    d = get_doc(session, AMENDMENT)
    assert d.regulator == "RBI"
    assert d.doc_type == "circular"
    assert d.circular_no == "DOR.HOL.REC.No.238/16.13.100/2026-27"
    assert d.rbi_no == "RBI/2026-27/278"
    assert d.department == "Department of Regulation"
    assert d.issued_date == d.valid_from == date(2026, 10, 1)
    assert d.valid_to is None
    assert d.url.endswith(f"NotificationUser.aspx?Id={AMENDMENT}&Mode=0")
    assert d.source_path == f"raw/rbi/notifications/{AMENDMENT}.html"
    assert "RBI/2026-27/278\nDOR.HOL.REC.No.238" in d.raw_text
    assert len(d.content_hash) == 64
    assert d.fetched_at is not None


def test_fema_notification_type(ingestor, session):
    ingestor.ingest([entry(FEMA)])
    d = get_doc(session, FEMA)
    assert d.doc_type == "notification"
    assert d.circular_no == "Notification No. FEMA 23(R)/(1)/2026-RB"


def test_rerun_is_idempotent_and_skips_fetching(ingestor, scraper):
    ingestor.ingest([entry(AMENDMENT), entry(OLD_MC)])
    stats = ingestor.ingest([entry(AMENDMENT), entry(OLD_MC)])
    assert (stats.new, stats.unchanged) == (0, 2)
    assert len(scraper.fetched) == 2  # only the first run fetched


def test_withdrawn_master_direction(ingestor, session):
    ingestor.ingest([entry(KYC_MD)], from_md_index=True)
    d = get_doc(session, KYC_MD)
    assert d.doc_type == "master_direction"
    assert d.updated_on == date(2025, 8, 14)
    assert d.withdrawn_on == d.valid_to == date(2025, 12, 4)
    assert d.valid_from == date(2016, 2, 25)


def test_newer_updated_stamp_refetches_and_rebuilds(ingestor, scraper, session, tmp_path):
    ingestor.ingest([entry(KYC_MD)], from_md_index=True)
    d = get_doc(session, KYC_MD)
    session.add(
        Chunk(
            doc_id=d.id,
            ord=0,
            text="old",
            char_start=0,
            char_end=3,
            embedding=[0.0] * EMBEDDING_DIM,
        )
    )
    session.commit()
    old_html = scraper.pages[KYC_MD]

    # Same stamp as stored: skipped without fetching.
    stats = ingestor.ingest([entry(KYC_MD, updated_on=date(2025, 8, 14))], from_md_index=True)
    assert stats.unchanged == 1 and len(scraper.fetched) == 1

    # Newer stamp and changed text: re-fetched, updated, chunks dropped, old HTML archived.
    scraper.pages[KYC_MD] = old_html.replace("Know Your Customer", "Know Your Client")
    stats = ingestor.ingest([entry(KYC_MD, updated_on=date(2026, 1, 9))], from_md_index=True)
    assert stats.updated == 1
    assert scraper.fetched[-1] == (KYC_MD, True)
    session.refresh(d)
    assert "Know Your Client" in d.raw_text
    assert session.scalars(select(Chunk).where(Chunk.doc_id == d.id)).all() == []
    history = list((tmp_path / "raw/rbi/notifications/history").glob(f"{KYC_MD}-*.html"))
    assert len(history) == 1 and history[0].read_text(encoding="utf-8") == old_html


def test_metadata_refreshes_when_text_is_unchanged(ingestor, scraper, session):
    ingestor.ingest([entry(AMENDMENT)])
    d = get_doc(session, AMENDMENT)
    d.department = None  # e.g. stored by an older parser
    session.add(
        Chunk(
            doc_id=d.id, ord=0, text="c", char_start=0, char_end=1, embedding=[0.0] * EMBEDDING_DIM
        )
    )
    session.commit()

    stats = ingestor.ingest([entry(AMENDMENT)], force=True)
    assert stats.updated == 1
    session.refresh(d)
    assert d.department == "Department of Regulation"
    assert len(session.scalars(select(Chunk).where(Chunk.doc_id == d.id)).all()) == 1  # kept


def test_reparse_uses_cache_without_fetching(ingestor, scraper, session):
    ingestor.ingest([entry(AMENDMENT), entry(FEMA)])
    get_doc(session, FEMA).department = None
    session.commit()
    fetched_before = len(scraper.fetched)

    stats = ingestor.reparse_cached()
    assert stats.updated == 1
    assert len(scraper.fetched) == fetched_before
    assert get_doc(session, FEMA).department == "Foreign Exchange Department"


def test_same_text_under_another_id_is_duplicate(ingestor, scraper):
    scraper.pages[OLD_MC] = scraper.pages[AMENDMENT]
    stats = ingestor.ingest([entry(AMENDMENT), entry(OLD_MC)])
    assert (stats.new, stats.duplicate) == (1, 1)


def test_fetch_failure_is_counted_and_run_continues(ingestor, scraper, session):
    scraper.errors[AMENDMENT] = FetchError("HTTP 500")
    stats = ingestor.ingest([entry(AMENDMENT), entry(OLD_MC)])
    assert (stats.failed, stats.new) == (1, 1)
    assert get_doc(session, OLD_MC).rbi_no == "RBI/2015-16/108"


def test_blocked_stops_the_run(ingestor, scraper):
    scraper.errors[AMENDMENT] = BlockedError("418")
    with pytest.raises(BlockedError):
        ingestor.ingest([entry(AMENDMENT), entry(OLD_MC)])
    assert [i for i, _ in scraper.fetched] == [AMENDMENT]


def test_md_index_marks_known_documents_as_master_directions(ingestor, session):
    ingestor.ingest([entry(OLD_MC)])
    assert get_doc(session, OLD_MC).doc_type == "circular"
    ingestor.ingest([entry(OLD_MC)], from_md_index=True)
    session.expire_all()
    assert get_doc(session, OLD_MC).doc_type == "master_direction"


def test_pdf_only_direction_takes_text_from_latest_version(ingestor, session):
    stats = ingestor.ingest([entry(PDF_ONLY, updated_on=date(2026, 10, 1))], from_md_index=True)
    assert stats.new == 1
    d = get_doc(session, PDF_ONLY)
    assert d.title == "Reserve Bank of India (Commercial Banks – Miscellaneous) Directions, 2025"
    assert d.doc_type == "master_direction"
    assert d.rbi_no == "RBI/DOR/2025-26/174"
    assert "Table of Contents" in d.raw_text
    assert d.updated_on == date(2026, 10, 1)  # RBI's current stamp, from the PDF-only page
    assert d.text_as_of == date(2025, 11, 28)  # but the text we hold is this version's
    assert d.source_path == f"raw/rbi/notifications/{PDF_ONLY}-v336.html"


def test_pdf_only_skips_newer_versions_that_are_pdf_only_too(ingestor, scraper, session):
    scraper.versions[PDF_ONLY].insert(0, ("detail_13136_pdf_only.html", 400, date(2026, 7, 1)))
    stats = ingestor.ingest([entry(PDF_ONLY, updated_on=date(2026, 10, 1))], from_md_index=True)
    assert stats.new == 1
    d = get_doc(session, PDF_ONLY)
    assert d.text_as_of == date(2025, 11, 28)
    assert d.source_path.endswith("-v336.html")


def test_pdf_only_when_no_version_has_text_fails(ingestor, scraper):
    scraper.versions[PDF_ONLY] = [("detail_13136_pdf_only.html", 400, date(2026, 7, 1))]
    assert ingestor.ingest([entry(PDF_ONLY)]).failed == 1


def test_pdf_only_without_versions_fails(ingestor, scraper, session):
    del scraper.versions[PDF_ONLY]
    stats = ingestor.ingest([entry(PDF_ONLY)])
    assert stats.failed == 1
    assert session.scalars(select(Document).where(Document.source_id == str(PDF_ONLY))).all() == []


def test_text_as_of_defaults_to_page_stamp_or_issue_date(ingestor, session):
    ingestor.ingest([entry(AMENDMENT), entry(KYC_MD)])
    assert get_doc(session, AMENDMENT).text_as_of == date(2026, 10, 1)  # issued
    assert get_doc(session, KYC_MD).text_as_of == date(2025, 8, 14)  # last "Updated as on"


def test_reparse_keeps_version_dates(ingestor, session):
    ingestor.ingest([entry(PDF_ONLY, updated_on=date(2026, 10, 1))], from_md_index=True)
    ingestor.reparse_cached()
    d = get_doc(session, PDF_ONLY)
    assert (d.updated_on, d.text_as_of) == (date(2026, 10, 1), date(2025, 11, 28))
