"""Thin HTTP client for the laptop backend (used by the Streamlit app)."""
import requests


class BackendOffline(Exception):
    """The backend (laptop + tunnel) could not be reached."""


class BackendClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 60):
        self.base_url = (base_url or "").rstrip("/")
        self.headers = {"X-API-Key": api_key or ""}
        self.timeout = timeout

    def _req(self, method: str, path: str, **kw) -> dict:
        if not self.base_url:
            raise BackendOffline("backend URL not configured")
        try:
            r = requests.request(method, self.base_url + path, headers=self.headers, timeout=self.timeout, **kw)
        except requests.RequestException as e:
            raise BackendOffline(str(e)) from e
        if r.status_code in (502, 503, 504, 530):  # tunnel/backend down
            raise BackendOffline(f"backend returned {r.status_code}")
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            return self._req("GET", "/health", ).get("status") == "ok"
        except Exception:
            return False

    def chat(self, session_id: str, message: str) -> dict:
        return self._req("POST", "/chat", json={"session_id": session_id, "message": message})

    def bookings(self) -> dict:
        return self._req("GET", "/bookings")
