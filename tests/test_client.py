import pytest

from frontend.client import BackendClient, BackendOffline

DEAD = "http://127.0.0.1:9"


def test_health_false_when_unreachable():
    assert BackendClient(DEAD, "k", timeout=2).health() is False


def test_chat_raises_backend_offline():
    with pytest.raises(BackendOffline):
        BackendClient(DEAD, "k", timeout=2).chat("s", "hi")


def test_empty_url_is_offline():
    assert BackendClient("", "k").health() is False
    with pytest.raises(BackendOffline):
        BackendClient("", "k").chat("s", "hi")


def test_health_uses_short_timeout(monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen["timeout"] = kw["timeout"]
        raise __import__("requests").ConnectionError("down")

    monkeypatch.setattr("frontend.client.requests.request", fake_request)
    assert BackendClient("https://x.trycloudflare.com", "k", timeout=120).health() is False
    assert seen["timeout"] <= 5
