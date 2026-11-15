from datetime import date, timedelta

import pytest

import config
from agent import tools
from backend import db

FUTURE = (date.today() + timedelta(days=10)).isoformat()
PAST = (date.today() - timedelta(days=1)).isoformat()


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    p = tmp_path / "bookings.db"
    monkeypatch.setattr(config, "DB_PATH", p)
    db.init_db(p)
    return p


@pytest.fixture
def no_rag(monkeypatch):
    monkeypatch.setattr(tools, "retrieve", lambda *a, **k: [])


def test_emi_known_value():
    out = tools.emi_calculator.invoke({"price": "5000000", "down_payment_pct": 30, "years": 5, "annual_rate_pct": 20})
    # loan 3.5m, r=20%/12, n=60 -> 92,729.x
    assert "Rs 1,500,000" in out and "Rs 3,500,000" in out
    assert "Rs 92,729" in out  # exact amortization: 92,728.59


def test_emi_accepts_lakh():
    out = tools.emi_calculator.invoke({"price": "50 lakh"})
    assert "Rs 3,500,000" in out


def test_emi_rejects_unparseable():
    assert tools.emi_calculator.invoke({"price": "expensive"}).startswith("Cannot calculate")


def test_booking_rejects_bad_phone():
    out = tools.book_test_drive.invoke({"name": "Ali", "phone": "12345", "model": "corolla", "city": "Lahore", "date": FUTURE})
    assert out.startswith("Cannot book") and "phone" in out
    assert db.list_bookings(config.DB_PATH) == []


def test_booking_rejects_past_date():
    out = tools.book_test_drive.invoke({"name": "Ali", "phone": "0300-1234567", "model": "corolla", "city": "Lahore", "date": PAST})
    assert out.startswith("Cannot book") and "date" in out


def test_booking_rejects_unknown_model():
    out = tools.book_test_drive.invoke({"name": "Ali", "phone": "03001234567", "model": "tesla", "city": "Lahore", "date": FUTURE})
    assert out.startswith("Cannot book") and "model" in out


def test_booking_saves_valid():
    out = tools.book_test_drive.invoke({"name": "Ali", "phone": "+92 300 1234567", "model": "corola", "city": "Lahore", "date": FUTURE})
    assert out.startswith("Booked #1")
    rows = db.list_bookings(config.DB_PATH)
    assert len(rows) == 1 and rows[0]["model"] == "Toyota Corolla" and rows[0]["phone"] == "+923001234567"


def test_service_booking_saves():
    out = tools.book_service.invoke({"name": "Sara", "phone": "03211234567", "model": "city", "km": 10000, "date": FUTURE})
    assert out.startswith("Booked #1")
    assert db.list_bookings(config.DB_PATH)[0]["kind"] == "service"


def test_compare_unknown_returns_refusal():
    assert tools.compare_models.invoke({"model_a": "corolla", "model_b": "tesla model 3"}) == config.REFUSAL


def test_compare_table_has_prices():
    out = tools.compare_models.invoke({"model_a": "corolla", "model_b": "civic"})
    assert "| Price" in out and "6,099,000" in out and "8,499,000" in out


def test_search_tesla_returns_refusal(no_rag):
    assert tools.search_cars.invoke({"query": "Tesla Model 3 price"}) == config.REFUSAL


def test_search_budget_filters(no_rag):
    out = tools.search_cars.invoke({"query": "family car", "max_price": "35 lakh"})
    assert "Alto" in out and "Corolla" not in out
    for line in out.splitlines():
        if "Rs " in line:
            price = int(line.split("Rs ")[1].split()[0].replace(",", "").rstrip(".,"))
            assert price <= 3_500_000


def test_search_by_model_name(no_rag):
    out = tools.search_cars.invoke({"query": "Honda Civic price"})
    assert "Civic" in out and "8,499,000" in out and "Source:" in out


def test_search_body_type(no_rag):
    out = tools.search_cars.invoke({"query": "suv", "body_type": "suv", "max_price": "1 crore"})
    assert "Oshan X7" in out


def test_ticket_created():
    out = tools.escalate_to_human.invoke({"summary": "Wants a discount on Fortuner"})
    assert out.startswith("Ticket #1 created")


def test_all_tools_count():
    assert len(tools.ALL_TOOLS) == 6


def test_suv_query_includes_crossovers(no_rag):
    out = tools.search_cars.invoke({"query": "hybrid suv under 1 crore"})
    assert "Corolla Cross" in out and "HR-V" in out


def test_search_query_schema_dict_falls_back_to_user_message(no_rag):
    token = tools.CURRENT_MESSAGE.set("Honda City price?")
    try:
        out = tools.search_cars.invoke({"query": {"type": "string"}, "brand": "Honda", "body_type": "sedan", "max_price": "1.2 crore"})
    finally:
        tools.CURRENT_MESSAGE.reset(token)
    assert "Honda City" in out and "4,737,000" in out


def test_search_ignores_invalid_brand(no_rag):
    out = tools.search_cars.invoke({"query": "best hybrid suv", "brand": "Hybrid", "max_price": "1.0 crore"})
    assert "Corolla Cross" in out


def test_search_named_model_ignores_wrong_body_type(no_rag):
    out = tools.search_cars.invoke({"query": "Alto ki qeemat", "brand": "Suzuki", "body_type": "sedan", "max_price": "50 lakh"})
    assert "Suzuki Alto" in out and "2,994,861" in out


def test_search_ignores_garbage_max_price(no_rag):
    out = tools.search_cars.invoke({"query": "Honda Civic", "max_price": {"type": "string"}})
    assert "8,499,000" in out


def test_emi_accepts_numeric_price():
    assert "Rs 3,500,000" in tools.emi_calculator.invoke({"price": 5000000})


def test_compare_accepts_non_string_gracefully():
    assert tools.compare_models.invoke({"model_a": {"type": "string"}, "model_b": "civic"}) == config.REFUSAL
