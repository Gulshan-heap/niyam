from fastapi.testclient import TestClient

from niyam.api.main import app, check_database

client = TestClient(app)


def test_health_ok_when_db_and_pgvector_present():
    app.dependency_overrides[check_database] = lambda: {"connected": True, "pgvector": True}
    try:
        resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_degraded_when_db_down():
    app.dependency_overrides[check_database] = lambda: {
        "connected": False,
        "error": "OperationalError",
    }
    try:
        resp = client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
