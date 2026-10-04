"""Daily ingestion worker: `python -m niyam.scheduler`.

Runs the RBI `recent` ingestion once at start-up (optional) and then every day at
NIYAM_INGEST_HOUR:NIYAM_INGEST_MINUTE India time.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session

from niyam.config import get_settings
from niyam.db.session import get_engine
from niyam.ingest.__main__ import build_ingestor, run_recent

log = logging.getLogger("niyam.scheduler")

TIMEZONE = "Asia/Kolkata"


def daily_ingest() -> None:
    with Session(get_engine()) as session:
        ingestor = build_ingestor(session)
        with ingestor.scraper.client:
            stats = run_recent(ingestor)
    log.info("daily RBI ingest done: %s", stats)


def build_scheduler() -> BlockingScheduler:
    s = get_settings()
    scheduler = BlockingScheduler(timezone=TIMEZONE)
    scheduler.add_job(
        daily_ingest,
        CronTrigger(hour=s.ingest_hour, minute=s.ingest_minute, timezone=TIMEZONE),
        id="rbi-daily",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    if s.ingest_on_start:
        scheduler.add_job(
            daily_ingest, id="rbi-startup", next_run_time=datetime.now(ZoneInfo(TIMEZONE))
        )
    return scheduler


def main() -> None:
    logging.basicConfig(
        level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    scheduler = build_scheduler()
    log.info("scheduler started: %s", [str(j.trigger) for j in scheduler.get_jobs()])
    scheduler.start()


if __name__ == "__main__":
    main()
