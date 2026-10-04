"""Polite HTTP client for regulator websites.

- Identifies itself with a descriptive User-Agent.
- Waits at least `min_interval` seconds between requests to the network.
- Retries connection errors, 429 and 5xx with exponential backoff (honouring Retry-After).
- Checks robots.txt once per host before fetching (standard semantics: 401/403 means
  disallow all, any other 4xx means no robots.txt so allow all, 5xx/unreachable means
  disallow all for now).
- Optionally caches responses on disk so re-runs don't hit the site again.
"""

import logging
import time
import urllib.robotparser
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class FetchError(Exception):
    pass


class BlockedError(FetchError):
    """The site refused us (firewall page or bot challenge). Never retried."""


class DisallowedError(FetchError):
    """robots.txt disallows this URL."""


def looks_blocked(resp: httpx.Response) -> bool:
    if resp.status_code in (403, 418):
        return True
    # F5 bot-challenge pages are served with 200 and a /TSPD/ script.
    return resp.status_code == 200 and "/TSPD/" in resp.text[:4000]


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        min_interval: float = 2.0,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.user_agent = user_agent
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.cache_dir = cache_dir
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_text(self, url: str, cache_key: str | None = None, refresh: bool = False) -> str:
        """GET a page as text. With `cache_key`, serve from / store to `cache_dir/cache_key`."""
        path = self._cache_path(cache_key)
        if path is not None and path.exists() and not refresh:
            return path.read_text(encoding="utf-8")
        text = self.request("GET", url).text
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        return text

    def post_text(self, url: str, data: dict[str, str]) -> str:
        return self.request("POST", url, data=data).text

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        self._check_robots(url)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise FetchError(f"{method} {url}: {exc!r}") from exc
                self._backoff(attempt, None, f"{type(exc).__name__}")
                continue
            if looks_blocked(resp):
                raise BlockedError(f"{method} {url}: blocked by site (HTTP {resp.status_code})")
            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._backoff(attempt, resp.headers.get("Retry-After"), f"HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                raise FetchError(f"{method} {url}: HTTP {resp.status_code}")
            return resp
        raise AssertionError("unreachable")

    def _cache_path(self, cache_key: str | None) -> Path | None:
        if cache_key is None or self.cache_dir is None:
            return None
        return self.cache_dir / cache_key

    def _throttle(self) -> None:
        now = self._clock()
        if self._last_request is not None:
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _backoff(self, attempt: int, retry_after: str | None, reason: str) -> None:
        delay = self.backoff_base * 2**attempt
        if retry_after and retry_after.isdigit():
            delay = max(delay, float(retry_after))
        log.warning("retrying after %s in %.1fs (attempt %d)", reason, delay, attempt + 1)
        self._sleep(delay)

    def _check_robots(self, url: str) -> None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots.get(origin)
        if rp is None:
            rp = self._load_robots(origin)
            self._robots[origin] = rp
        if not rp.can_fetch(self.user_agent, url):
            raise DisallowedError(f"robots.txt disallows {url}")

    def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser:
        rp = urllib.robotparser.RobotFileParser(origin + "/robots.txt")
        self._throttle()
        try:
            resp = self._client.get(origin + "/robots.txt")
        except httpx.TransportError:
            rp.disallow_all = True
            log.warning("robots.txt unreachable for %s; treating as disallow-all", origin)
            return rp
        if resp.status_code in (401, 403):
            rp.disallow_all = True
        elif 400 <= resp.status_code < 500:
            rp.allow_all = True
            log.info("no robots.txt for %s (HTTP %d)", origin, resp.status_code)
        elif resp.status_code >= 500:
            rp.disallow_all = True
        else:
            rp.parse(resp.text.splitlines())
        return rp
