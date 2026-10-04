"""Turn RBI listing entries into `documents` rows.

Idempotent: documents already stored are skipped without fetching, unless the listing shows
a newer "Updated as on" stamp (Master Directions are edited in place). When a document's text
changes, its chunks are deleted so they get rebuilt, and the previous HTML is archived.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from niyam.db.models import Chunk, Document
from niyam.ingest.http import BlockedError, FetchError
from niyam.ingest.scrapers.rbi import (
    DetailPage,
    ListingEntry,
    PageNotFoundError,
    RbiScraper,
    detail_url,
    is_master_direction,
    parse_detail,
)

log = logging.getLogger(__name__)

REGULATOR = "RBI"
VERSION_FILE_RE = re.compile(r"-v\d+\.html$")  # cached dated version, e.g. 13136-v336.html
MAX_VERSIONS_TRIED = 4


@dataclass
class IngestStats:
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicate: int = 0
    missing: int = 0  # no document at that id
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
    fetched: bool = True,
    text_as_of: date | None = None,
) -> str:
    """Insert or update one document. Returns new / updated / unchanged / duplicate.

    Metadata is always re-derived from the page (so parser fixes apply on re-runs); chunks
    are only dropped when the text itself changed. `text_as_of` defaults to the page's own
    "Updated as on" stamp or issue date.
    """
    if not page.has_text:
        raise ValueError(f"RBI {page.rbi_id}: page has no text")
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
    text_changed = doc is None or doc.content_hash != digest
    if text_changed:
        clash = session.scalar(
            select(Document.source_id).where(
                Document.content_hash == digest, Document.id != (doc.id if doc else -1)
            )
        )
        if clash is not None:
            log.info("RBI %s has the same text as RBI %s; skipping", page.rbi_id, clash)
            return "duplicate"

    is_md = from_md_index or (doc is not None and doc.doc_type == "master_direction")
    values = {
        "circular_no": page.ref_no or page.rbi_no,
        "rbi_no": page.rbi_no,
        "title": page.title,
        "department": page.department,
        "doc_type": doc_type_for(page, is_md),
        "issued_date": issued,
        "updated_on": page.updated_on,
        "withdrawn_on": page.withdrawn_on,
        "text_as_of": text_as_of or page.updated_on or issued,
        # Initial validity window from what the page states; the temporal layer refines it.
        "valid_from": (doc.effective_from if doc else None) or issued,
        "valid_to": page.withdrawn_on,
        "url": detail_url(page.rbi_id),
        "pdf_url": page.pdf_url or pdf_url,
        "source_path": source_path,
        "raw_text": text,
        "content_hash": digest,
    }

    if doc is None:
        outcome = "new"
        doc = Document(regulator=REGULATOR, source_id=str(page.rbi_id))
        session.add(doc)
    elif text_changed:
        outcome = "updated"
        session.execute(delete(Chunk).where(Chunk.doc_id == doc.id))  # rebuilt from new text
    elif any(getattr(doc, k) != v for k, v in values.items()):
        outcome = "updated"
    else:
        return "unchanged"

    for k, v in values.items():
        setattr(doc, k, v)
    if fetched:
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

    def reparse_cached(self) -> IngestStats:
        """Re-derive every stored document from its cached HTML. No network."""
        stats = IngestStats()
        docs = self.session.execute(
            select(
                Document.source_id,
                Document.source_path,
                Document.issued_date,
                Document.updated_on,
                Document.text_as_of,
            ).where(Document.regulator == REGULATOR, Document.source_path.is_not(None))
        ).all()
        for source_id, source_path, issued, updated_on, text_as_of in docs:
            path = Path(source_path)
            if not path.is_absolute():
                path = self.data_dir / path
            from_version = VERSION_FILE_RE.search(path.name) is not None
            try:
                page = parse_detail(path.read_text(encoding="utf-8"), int(source_id))
                if from_version:
                    # A dated version page doesn't carry the current "Updated as on" stamp.
                    page.updated_on = updated_on
                outcome = upsert_document(
                    self.session,
                    page,
                    listed_date=issued,
                    source_path=source_path,
                    fetched=False,
                    text_as_of=text_as_of if from_version else None,
                )
            except (OSError, ValueError) as exc:
                self.session.rollback()
                log.warning("RBI %s reparse failed: %s", source_id, exc)
                stats.add("failed")
                continue
            self.session.commit()
            stats.add(outcome)
        return stats

    def _ingest_one(self, entry: ListingEntry, *, from_md_index: bool, refresh: bool) -> str:
        old_html = None
        cached = self._cache_file(entry.rbi_id)
        if refresh and cached is not None and cached.exists():
            old_html = cached.read_text(encoding="utf-8")

        html, path = self.scraper.fetch_detail(entry.rbi_id, refresh=refresh)
        try:
            page = parse_detail(html, entry.rbi_id)
        except PageNotFoundError:
            if path is not None:
                path.unlink(missing_ok=True)  # don't cache: the id may be used later
            return "missing"
        text_as_of = None
        if not page.has_text and page.has_previous_versions:
            page, path, text_as_of = self._latest_version(page)
        if not page.title:
            page.title = entry.title
        outcome = upsert_document(
            self.session,
            page,
            listed_date=entry.listed_date,
            pdf_url=entry.pdf_url,
            source_path=self._relative(path),
            from_md_index=from_md_index,
            text_as_of=text_as_of,
        )
        if outcome == "updated" and old_html and old_html != html and cached is not None:
            self._archive(cached, old_html)
        log.info("RBI %s %s: %s", entry.rbi_id, outcome, page.title[:80])
        return outcome

    def _latest_version(self, current: DetailPage) -> tuple[DetailPage, Path | None, date]:
        """The current page is PDF-only: take the text of the newest dated version that has
        HTML text (recent versions can be PDF-only too), but keep the current page's title,
        "Updated as on" stamp and PDF link."""
        versions = self.scraper.list_versions(current.rbi_id)
        for version in versions[:MAX_VERSIONS_TRIED]:
            html, path = self.scraper.fetch_version(current.rbi_id, version)
            page = parse_detail(html, current.rbi_id)
            if not page.has_text:
                continue
            page.title = current.title or page.title
            page.updated_on = current.updated_on
            page.pdf_url = current.pdf_url or page.pdf_url
            log.info(
                "RBI %s is PDF-only; using text of version dated %s", current.rbi_id, version.as_of
            )
            return page, path, version.as_of
        raise ValueError(
            f"RBI {current.rbi_id}: PDF-only, and none of {len(versions)} versions has HTML text"
        )

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
