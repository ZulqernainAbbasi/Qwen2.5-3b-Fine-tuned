import pytest

from agent.parsing import match_model, parse_price


@pytest.mark.parametrize("text,expected", [
    ("under 50 lakh", 5_000_000),
    ("1.2 crore", 12_000_000),
    ("45 lac", 4_500_000),
    ("6 million", 6_000_000),
    ("Rs 7,499,000", 7_499_000),
    ("80 lakh tak", 8_000_000),
    ("budget 3.5m", 3_500_000),
    ("cheap", None),
    ("", None),
])
def test_parse_price(text, expected):
    assert parse_price(text) == expected


@pytest.mark.parametrize("name,expected", [
    ("corola", ("Toyota", "Corolla")),
    ("civic rs", ("Honda", "Civic")),
    ("alto ki qeemat", ("Suzuki", "Alto")),
    ("Corolla Cross hybrid", ("Toyota", "Corolla Cross")),
    ("oshan x7", ("Changan", "Oshan X7")),
    ("sportage", ("KIA", "Sportage L")),
    ("tesla model 3", None),
    ("mercedes c200", None),
])
def test_match_model(name, expected):
    assert match_model(name) == expected
