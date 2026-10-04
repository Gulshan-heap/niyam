"""Turn RBI listing entries into `documents` rows.

Idempotent: documents already stored are skipped without fetching, unless the listing shows
a newer "Updated as on" stamp (Master Directions are edited in place). When a document's text
changes, its chunks are deleted so they get rebuilt, and the previous HTML is archived.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from niyam.db.models import Chunk, Document
from niyam.ingest.http import BlockedError, FetchError
from niyam.ingest.scrapers.rbi import (
    DetailPage,
    ListingEntry,
    RbiScraper,
    detail_url,
    is_master_direction,
    parse_detail,
)

log = logging.getLogger(__name__)

REGULATOR = "RBI"


@dataclass
class IngestStats:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicate: int = 0
    failed: int = 0

    def add(self, outcome: str) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)

    def __str__(self) -> str:
        return ", ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


def doc_type_for(page: DetailPage, from_md_index: bool) -> str:
    if from_md_index or is_master_direction(page.title):
        return "master_direction"
    if page.ref_no and re.search(r"\bNotification\b", page.ref_no, re.I):
        return "notification"
    return "circular"


def upsert_document(
    session: Session,
    page: DetailPage,
    *,
    listed_date=None,
    pdf_url: str | None = None,
    source_path: str | None = None,
    from_md_index: bool = False,
) -> str:
    """Insert or update one document. Returns new / updated / unchanged / duplicate."""
    text = page.text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    issued = page.issued_date or listed_date
    if issued is None:
        raise ValueError(f"RBI {page.rbi_id}: no issue date on page or listing")

    doc = session.scalar(
        select(Document).where(
            Document.regulator == REGULATOR, Document.source_id == str(page.rbi_id)
        )
    )
    if doc is not None and doc.content_hash == digest:
        return "unchanged"
    clash = session.scalar(
        select(Document.source_id).where(
            Document.content_hash == digest, Document.id != (doc.id if doc else -1)
        )
    )
    if clash is not None:
        log.info("RBI %s has the same text as RBI %s; skipping", page.rbi_id, clash)
        return "duplicate"

    if doc is None:
        outcome = "new"
        doc = Document(regulator=REGULATOR, source_id=str(page.rbi_id))
        session.add(doc)
    else:
        outcome = "updated"
        session.execute(delete(Chunk).where(Chunk.doc_id == doc.id))  # rebuilt from new text

    doc.circular_no = page.ref_no or page.rbi_no
    doc.rbi_no = page.rbi_no
    doc.title = page.title
    doc.department = page.department
    doc.doc_type = doc_type_for(page, from_md_index)
    doc.issued_date = issued
    doc.updated_on = page.updated_on
    doc.withdrawn_on = page.withdrawn_on
    # Initial validity window from what the page states; the temporal layer refines it.
    doc.valid_from = doc.effective_from or issued
    doc.valid_to = page.withdrawn_on
    doc.url = detail_url(page.rbi_id)
    doc.pdf_url = page.pdf_url or pdf_url
    doc.source_path = source_path
    doc.raw_text = text
    doc.content_hash = digest
    doc.fetched_at = datetime.now(UTC)
    session.flush()
    return outcome


class RbiIngestor:
    def __init__(self, session: Session, scraper: RbiScraper, data_dir: Path):
        self.session = session
        self.scraper = scraper
        self.data_dir = data_dir

    def ingest(
        self,
        entries: list[ListingEntry],
        *,
        from_md_index: bool = False,
        force: bool = False,
    ) -> IngestStats:
        stats = IngestStats()
        known = dict(
            self.session.execute(
                select(Document.source_id, Document.updated_on).where(
                    Document.regulator == REGULATOR,
                    Document.source_id.in_([str(e.rbi_id) for e in entries]),
                )
            ).all()
        )
        for entry in entries:
            refresh = force
            if str(entry.rbi_id) in known and not force:
                stored = known[str(entry.rbi_id)]
                if entry.updated_on is None or (stored and entry.updated_on <= stored):
                    stats.add("unchanged")
                    continue
                refresh = True  # listing shows a newer "Updated as on" stamp
            try:
                outcome = self._ingest_one(entry, from_md_index=from_md_index, refresh=refresh)
            except BlockedError:
                self.session.rollback()
                log.error("site is blocking us; stopping this run")
                raise
            except (FetchError, ValueError) as exc:
                self.session.rollback()
                log.warning("RBI %s failed: %s", entry.rbi_id, exc)
                stats.add("failed")
                continue
            self.session.commit()
            stats.add(outcome)
            log.info("RBI %s %s: %s", entry.rbi_id, outcome, entry.title[:80])
        if from_md_index:
            # Documents first seen in a monthly listing may only now be known to be MDs.
            self.session.execute(
                update(Document)
                .where(
                    Document.regulator == REGULATOR,
                    Document.source_id.in_([str(e.rbi_id) for e in entries]),
                )
                .values(doc_type="master_direction")
            )
            self.session.commit()
        return stats

    def _ingest_one(self, entry: ListingEntry, *, from_md_index: bool, refresh: bool) -> str:
        old_html = None
        cached = self._cache_file(entry.rbi_id)
        if refresh and cached is not None and cached.exists():
            old_html = cached.read_text(encoding="utf-8")

        html, path = self.scraper.fetch_detail(entry.rbi_id, refresh=refresh)
        page = parse_detail(html, entry.rbi_id)
        if not page.title:
            page.title = entry.title
        outcome = upsert_document(
            self.session,
            page,
            listed_date=entry.listed_date,
            pdf_url=entry.pdf_url,
            source_path=self._relative(path),
            from_md_index=from_md_index,
        )
        if outcome == "updated" and old_html and old_html != html and cached is not None:
            self._archive(cached, old_html)
        return outcome

    def _cache_file(self, rbi_id: int) -> Path | None:
        root = self.scraper.client.cache_dir
        return root / f"rbi/notifications/{rbi_id}.html" if root else None

    def _archive(self, cached: Path, old_html: str) -> None:
        digest = hashlib.sha256(old_html.encode("utf-8")).hexdigest()[:12]
        dest = cached.parent / "history" / f"{cached.stem}-{digest}.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(old_html, encoding="utf-8")

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.data_dir.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
