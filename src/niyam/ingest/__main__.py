"""Ingestion CLI.

    python -m niyam.ingest recent                      # this month + last month (daily job)
    python -m niyam.ingest months 2024-01 2024-06      # backfill a range of months
    python -m niyam.ingest master-directions           # the Master Directions index
    python -m niyam.ingest ids 12200 13725             # every page id in a range
    python -m niyam.ingest reparse                     # re-derive stored docs from cached HTML
    python -m niyam.ingest index                       # chunk new documents, embed new chunks
    python -m niyam.ingest index --no-embed            # chunk only (seconds); embed later
    python -m niyam.ingest relations                   # rebuild amends/supersedes/... links

Common options: --match REGEX (filter titles), --limit N, --force (re-fetch known documents).

The monthly listings omit many past circulars (withdrawn ones among them), so `ids` is the
way to get complete history: RBI page ids are sequential.
"""

import argparse
import logging
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from niyam.config import get_settings
from niyam.db.session import get_engine
from niyam.ingest.http import PoliteClient
from niyam.ingest.index import build_index
from niyam.ingest.pipeline import IngestStats, RbiIngestor
from niyam.ingest.relations import extract_all, save_relations
from niyam.ingest.scrapers.rbi import ListingEntry, RbiScraper
from niyam.retrieval.embeddings import get_embedder

log = logging.getLogger("niyam.ingest")


def month_range(start: str, end: str) -> Iterator[tuple[int, int]]:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def previous_month(today: date) -> tuple[int, int]:
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)


def select_entries(
    entries: list[ListingEntry], match: str | None = None, section: str | None = None
) -> list[ListingEntry]:
    if match:
        entries = [e for e in entries if re.search(match, e.title, re.I)]
    if section:
        entries = [e for e in entries if e.section and re.search(section, e.section, re.I)]
    return entries


def id_entries(first: int, last: int) -> list[ListingEntry]:
    """Entries for a page-id range, newest first. Titles and dates come from the pages."""
    return [ListingEntry(i, "", None, None) for i in range(last, first - 1, -1)]


def run_recent(ingestor: RbiIngestor, today: date | None = None) -> IngestStats:
    """Daily job: re-list this month and last month (late postings), plus MD updates."""
    today = today or date.today()
    total = IngestStats()
    entries: list[ListingEntry] = []
    for y, m in [previous_month(today), (today.year, today.month)]:
        entries += ingestor.scraper.list_month(y, m)
    _merge(total, ingestor.ingest(entries))
    _merge(total, ingestor.ingest(ingestor.scraper.list_master_directions(), from_md_index=True))
    return total


def _merge(total: IngestStats, part: IngestStats) -> None:
    for name, value in vars(part).items():
        setattr(total, name, getattr(total, name) + value)


def build_ingestor(session: Session) -> RbiIngestor:
    s = get_settings()
    data_dir = Path(s.data_dir)
    client = PoliteClient(
        user_agent=s.user_agent,
        min_interval=s.request_interval_seconds,
        timeout=s.http_timeout_seconds,
        cache_dir=data_dir / "raw",
    )
    return RbiIngestor(session, RbiScraper(client), data_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m niyam.ingest")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("recent", help="this month and last month, plus Master Direction updates")
    months = sub.add_parser("months", help="backfill notifications for a range of months")
    months.add_argument("start", help="YYYY-MM")
    months.add_argument("end", nargs="?", help="YYYY-MM (default: same as start)")
    md = sub.add_parser("master-directions", help="ingest the Master Directions index")
    md.add_argument("--section", help="regex on the index section, e.g. 'Non-Banking'")
    ids = sub.add_parser("ids", help="ingest every page id in a range (complete history)")
    ids.add_argument("first", type=int)
    ids.add_argument("last", type=int, help="inclusive")
    ids.add_argument("--limit", type=int, help="ingest at most N documents")
    ids.add_argument("--force", action="store_true", help="re-fetch documents already stored")
    sub.add_parser("reparse", help="re-run the parser on cached HTML (no network)")
    sub.add_parser("relations", help="rebuild relations between documents (no network)")
    index = sub.add_parser("index", help="chunk and embed documents that have no chunks yet")
    index.add_argument("--limit", type=int, help="chunk at most N documents")
    index.add_argument("--no-embed", action="store_true", help="chunk only, skip embeddings")
    for p in (months, md):
        p.add_argument("--match", help="regex on titles, e.g. 'KYC|digital lending'")
        p.add_argument("--limit", type=int, help="ingest at most N documents")
        p.add_argument("--force", action="store_true", help="re-fetch documents already stored")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    with Session(get_engine()) as session:
        ingestor = build_ingestor(session)
        with ingestor.scraper.client:
            if args.cmd == "recent":
                stats = run_recent(ingestor)
            elif args.cmd == "reparse":
                stats = ingestor.reparse_cached()
            elif args.cmd == "relations":
                found = extract_all(session, ingestor.data_dir)
                stats = f"{save_relations(session, found)} relations saved"
            elif args.cmd == "index":
                embedder = None if args.no_embed else get_embedder()
                stats = build_index(session, embedder, limit=args.limit)
            elif args.cmd == "ids":
                entries = id_entries(args.first, args.last)[: args.limit]
                stats = ingestor.ingest(entries, force=args.force)
            elif args.cmd == "months":
                stats = IngestStats()
                remaining = args.limit
                for y, m in month_range(args.start, args.end or args.start):
                    entries = select_entries(ingestor.scraper.list_month(y, m), args.match)
                    if remaining is not None:
                        entries = entries[:remaining]
                        remaining -= len(entries)
                    log.info("%d-%02d: %d entries", y, m, len(entries))
                    _merge(stats, ingestor.ingest(entries, force=args.force))
                    if remaining == 0:
                        break
            else:
                entries = select_entries(
                    ingestor.scraper.list_master_directions(), args.match, args.section
                )[: args.limit]
                stats = ingestor.ingest(entries, from_md_index=True, force=args.force)
    log.info("done: %s", stats)


if __name__ == "__main__":
    main()
