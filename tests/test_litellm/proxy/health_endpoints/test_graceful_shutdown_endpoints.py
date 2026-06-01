"""
Behaviour tests for the graceful-shutdown health probes.

Builds a minimal FastAPI app from the health router plus
InFlightRequestsMiddleware so the probe responses can be asserted without
standing up the full proxy.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.health_endpoints._health_endpoints import router
from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager


@pytest.fixture(autouse=True)
def _reset():
    GracefulShutdownManager.reset()
    InFlightRequestsMiddleware._in_flight = 0
    yield
    GracefulShutdownManager.reset()
    InFlightRequestsMiddleware._in_flight = 0


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(InFlightRequestsMiddleware)
    return TestClient(app)


@pytest.fixture
def enable_drain(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server, "general_settings", {"enable_drain_endpoint": True}
    )


def test_drain_disabled_by_default_returns_404_with_no_side_effect(client, monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {})
    resp = client.get("/health/drain")
    assert resp.status_code == 404
    assert GracefulShutdownManager.is_shutting_down() is False


def test_drain_when_enabled_sets_shutting_down_and_returns_drained(
    client, enable_drain
):
    resp = client.get("/health/drain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "drained"
    assert body["drained_requests"] == 0
    assert GracefulShutdownManager.is_shutting_down() is True


def test_readiness_returns_503_shutting_down_during_drain(client):
    GracefulShutdownManager.start_shutdown()
    resp = client.get("/health/readiness")
    assert resp.status_code == 503
    assert resp.json() == {"status": "shutting_down"}


def test_readiness_does_not_report_shutting_down_normally(client):
    resp = client.get("/health/readiness")
    assert resp.json().get("status") != "shutting_down"


def test_liveliness_returns_503_during_drain(client):
    GracefulShutdownManager.start_shutdown()
    resp = client.get("/health/liveliness")
    assert resp.status_code == 503
    assert resp.json() == {"status": "shutting_down"}


def test_liveness_alias_returns_503_during_drain(client):
    GracefulShutdownManager.start_shutdown()
    resp = client.get("/health/liveness")
    assert resp.status_code == 503


def test_liveliness_returns_alive_when_not_shutting_down(client):
    resp = client.get("/health/liveliness")
    assert resp.status_code == 200
    assert resp.json() == "I'm alive!"
