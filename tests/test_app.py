"""Render the Streamlit app headlessly with the backend offline; it must not crash."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_renders_offline(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "")
    at = st_testing.AppTest.from_file(str(__import__("config").ROOT / "frontend" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("offline" in w.value.lower() for w in at.warning)
    assert len(at.tabs) == 4


def test_showroom_filter_runs(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "")
    at = st_testing.AppTest.from_file(str(__import__("config").ROOT / "frontend" / "app.py"), default_timeout=60).run()
    at.selectbox[0].set_value("suv").run()
    assert not at.exception
