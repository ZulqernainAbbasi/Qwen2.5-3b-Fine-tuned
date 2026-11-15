import pytest
from fastapi.testclient import TestClient

import config
from backend import api, db

KEY = "test-key"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CAR_API_KEY", KEY)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "b.db")
    monkeypatch.setattr(api, "run_agent", lambda sid, msg: {"reply": f"echo:{msg}", "steps": [{"tool": "search_cars", "args": {}, "result": "x"}]})
    return TestClient(api.app)


def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model"] == config.LLM_MODEL


def test_chat_requires_key(client):
    assert client.post("/chat", json={"session_id": "s", "message": "hi"}).status_code == 401
    assert client.post("/chat", json={"session_id": "s", "message": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_chat_returns_reply_and_steps(client):
    r = client.post("/chat", json={"session_id": "s", "message": "hi"}, headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert r.json() == {"reply": "echo:hi", "steps": [{"tool": "search_cars", "args": {}, "result": "x"}]}


def test_empty_message_422(client):
    r = client.post("/chat", json={"session_id": "s", "message": ""}, headers={"X-API-Key": KEY})
    assert r.status_code == 422


def test_too_long_message_422(client):
    r = client.post("/chat", json={"session_id": "s", "message": "a" * 1001}, headers={"X-API-Key": KEY})
    assert r.status_code == 422


def test_bookings_lists(client):
    db.add_booking("test_drive", name="Ali", phone="+923001234567", model="Toyota Corolla", city="Lahore", date="2030-01-01")
    db.add_ticket("help")
    assert client.get("/bookings").status_code == 401
    r = client.get("/bookings", headers={"X-API-Key": KEY})
    assert len(r.json()["bookings"]) == 1 and len(r.json()["tickets"]) == 1


def test_agent_error_returns_503(client, monkeypatch):
    def boom(sid, msg):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(api, "run_agent", boom)
    r = client.post("/chat", json={"session_id": "s", "message": "hi"}, headers={"X-API-Key": KEY})
    assert r.status_code == 503


def test_server_refuses_when_key_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("CAR_API_KEY", raising=False)
    c = TestClient(api.app)
    assert c.post("/chat", json={"session_id": "s", "message": "hi"}, headers={"X-API-Key": ""}).status_code == 401
