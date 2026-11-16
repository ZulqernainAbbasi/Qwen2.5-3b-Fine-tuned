"""Render the Streamlit app headlessly with the backend offline; it must not crash."""
from pathlib import Path

import pytest

import config

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(config.ROOT / "frontend" / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "")
    return st_testing.AppTest.from_file(APP, default_timeout=60).run()


def test_app_renders_offline(app):
    assert not app.exception
    assert any("offline" in w.value.lower() for w in app.warning)
    assert len(app.tabs) == 5


def test_no_car_emoji_in_frontend():
    src = Path(APP).read_text(encoding="utf-8")
    for emoji in ["🚗", "🚘", "🚙", "🏎"]:
        assert emoji not in src


def test_financing_tab_quotes_corolla_with_sbp_rules(app):
    car = next(s for s in app.selectbox if s.label == "Car")
    car.set_value("Toyota Corolla").run()
    assert not app.exception
    notes = " ".join(w.value for w in list(app.warning) + list(app.info))
    assert "3,000,000" in notes  # SBP cap forces a bigger down payment


def test_compare_tab_switch(app):
    first = next(s for s in app.selectbox if s.label == "First car")
    first.set_value("Suzuki Alto").run()
    assert not app.exception
