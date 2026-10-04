import httpx
import pytest

from niyam.ingest.http import BlockedError, DisallowedError, FetchError, PoliteClient


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s

    def clock(self) -> float:
        return self.now


def make_client(handler, tmp_path=None, min_interval=2.0, **kw) -> tuple[PoliteClient, FakeTime]:
    t = FakeTime()
    client = PoliteClient(
        user_agent="TestBot/1.0",
        min_interval=min_interval,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=t.sleep,
        clock=t.clock,
        **kw,
    )
    return client, t


def no_robots(handler):
    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return handler(request)

    return wrapped


def test_throttles_between_requests():
    client, t = make_client(no_robots(lambda r: httpx.Response(200, text="ok")))
    client.get_text("https://x.test/a")
    client.get_text("https://x.test/b")
    # robots.txt, /a, /b: every request after the first waits the full interval
    assert t.sleeps == [2.0, 2.0]


def test_sends_user_agent():
    seen = []

    def handler(r):
        seen.append(r.headers["user-agent"])
        return httpx.Response(200, text="ok")

    client, _ = make_client(no_robots(handler))
    client.get_text("https://x.test/a")
    assert seen == ["TestBot/1.0"]


def test_retries_5xx_then_succeeds():
    calls = iter([httpx.Response(503), httpx.Response(502), httpx.Response(200, text="ok")])
    client, t = make_client(no_robots(lambda r: next(calls)), min_interval=0, backoff_base=1.0)
    assert client.get_text("https://x.test/a") == "ok"
    assert t.sleeps == [1.0, 2.0]  # exponential backoff


def test_honours_retry_after():
    calls = iter([httpx.Response(429, headers={"Retry-After": "30"}), httpx.Response(200)])
    client, t = make_client(no_robots(lambda r: next(calls)), min_interval=0)
    client.get_text("https://x.test/a")
    assert t.sleeps == [30.0]


def test_gives_up_after_max_retries():
    client, _ = make_client(no_robots(lambda r: httpx.Response(500)), max_retries=2)
    with pytest.raises(FetchError):
        client.get_text("https://x.test/a")


def test_transport_errors_are_retried():
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, text="ok")

    client, _ = make_client(no_robots(handler))
    assert client.get_text("https://x.test/a") == "ok"


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(418, text="Unauthorised Access"),
        httpx.Response(403),
        httpx.Response(200, text='<script>window["bobcmn"]="1011/TSPD/3000TSPD_101"</script>'),
    ],
)
def test_blocked_pages_raise_without_retry(resp):
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return resp

    client, _ = make_client(no_robots(handler))
    with pytest.raises(BlockedError):
        client.get_text("https://x.test/a")
    assert calls["n"] == 1


def test_robots_disallow_is_respected():
    def handler(r):
        if r.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="ok")

    client, _ = make_client(handler)
    assert client.get_text("https://x.test/public") == "ok"
    with pytest.raises(DisallowedError):
        client.get_text("https://x.test/private/page")


def test_robots_fetched_once_per_host():
    robots_calls = []

    def handler(r):
        if r.url.path == "/robots.txt":
            robots_calls.append(r.url.host)
            return httpx.Response(404)
        return httpx.Response(200, text="ok")

    client, _ = make_client(handler)
    client.get_text("https://x.test/a")
    client.get_text("https://x.test/b")
    client.get_text("https://y.test/a")
    assert robots_calls == ["x.test", "y.test"]


def test_robots_403_means_disallow_all():
    def handler(r):
        if r.url.path == "/robots.txt":
            return httpx.Response(403)
        return httpx.Response(200, text="ok")

    client, _ = make_client(handler)
    with pytest.raises(DisallowedError):
        client.get_text("https://x.test/a")


def test_cache_avoids_second_request(tmp_path):
    calls = {"n": 0}

    def handler(r):
        calls["n"] += 1
        return httpx.Response(200, text="page ₹ body")

    client, _ = make_client(no_robots(handler), tmp_path)
    assert client.get_text("https://x.test/a", cache_key="rbi/a.html") == "page ₹ body"
    assert client.get_text("https://x.test/a", cache_key="rbi/a.html") == "page ₹ body"
    assert calls["n"] == 1
    assert (tmp_path / "rbi" / "a.html").exists()

    client.get_text("https://x.test/a", cache_key="rbi/a.html", refresh=True)
    assert calls["n"] == 2
