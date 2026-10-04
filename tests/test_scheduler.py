from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from niyam.config import get_settings
from niyam.scheduler import build_scheduler, daily_ingest


def test_daily_job_runs_in_india_time(monkeypatch):
    monkeypatch.setenv("NIYAM_INGEST_HOUR", "21")
    monkeypatch.setenv("NIYAM_INGEST_ON_START", "false")
    get_settings.cache_clear()
    try:
        jobs = build_scheduler().get_jobs()
    finally:
        get_settings.cache_clear()
    assert [j.id for j in jobs] == ["rbi-daily"]
    job = jobs[0]
    assert job.func is daily_ingest
    assert str(job.trigger.timezone) == "Asia/Kolkata"
    assert "hour='21'" in str(job.trigger)


def test_startup_run_is_optional(monkeypatch):
    monkeypatch.setenv("NIYAM_INGEST_ON_START", "true")
    get_settings.cache_clear()
    try:
        ids = [j.id for j in build_scheduler().get_jobs()]
    finally:
        get_settings.cache_clear()
    assert ids == ["rbi-daily", "rbi-startup"]


def test_startup_run_is_due_now_regardless_of_host_timezone(monkeypatch):
    # A naive datetime.now() on a UTC host would be read as IST, i.e. 5.5h in the past.
    monkeypatch.setenv("NIYAM_INGEST_ON_START", "true")
    get_settings.cache_clear()
    try:
        job = next(j for j in build_scheduler().get_jobs() if j.id == "rbi-startup")
    finally:
        get_settings.cache_clear()
    run_at = job.trigger.run_date
    assert run_at.tzinfo is not None
    assert abs(run_at - datetime.now(ZoneInfo("UTC"))) < timedelta(minutes=1)
