import pytest

from agent.finance import emi, load_banks, quote, sbp_terms


def test_sbp_terms_small_car():
    t = sbp_terms(2_994_861, 658)
    assert t["max_years"] == 5 and t["min_down_pct"] == 30 and t["max_loan"] == 3_000_000


def test_sbp_terms_big_car_and_ev():
    assert sbp_terms(6_099_000, 1598)["max_years"] == 3
    assert sbp_terms(6_949_000, None)["max_years"] == 3  # EV: assume >1000cc rule


def test_quote_small_car_within_rules():
    q = quote(2_994_861, 658, years=5, down_pct=30)
    assert q["down"] == 898_458 and q["loan"] == 2_096_403 and q["years"] == 5
    assert q["notes"] == []


def test_quote_raises_down_payment_to_respect_cap():
    q = quote(6_099_000, 1598, years=3, down_pct=30)
    assert q["loan"] == 3_000_000 and q["down"] == 3_099_000
    assert any("Rs 3,000,000" in n for n in q["notes"])


def test_quote_caps_years_by_engine():
    q = quote(6_099_000, 1598, years=5, down_pct=60)
    assert q["years"] == 3 and any("3 years" in n for n in q["notes"])


def test_quote_raises_down_below_30():
    q = quote(4_000_000, 998, years=3, down_pct=10)
    assert q["down"] == 1_200_000 and any("30%" in n for n in q["notes"])


def test_bank_rows_ordered_and_consistent():
    q = quote(4_000_000, 998, years=5, down_pct=30)
    rows = q["banks"]
    assert len(rows) == len(load_banks()) >= 6
    for r in rows:
        assert r["rate_low"] <= r["rate_high"] and r["emi_low"] <= r["emi_high"]
        assert r["total_low"] == round(r["emi_low"] * 60)
    assert [r["emi_low"] for r in rows] == sorted(r["emi_low"] for r in rows)


def test_emi_formula():
    assert round(emi(3_500_000, 20, 5)) == 92_729
    assert emi(1_200_000, 0, 1) == 100_000


def test_invalid_years():
    with pytest.raises(ValueError):
        quote(4_000_000, 998, years=0)


def test_by_years_table():
    from agent.finance import emi_by_years
    rows = emi_by_years(2_994_861, 658, down_pct=30, rate_pct=15)
    assert [r["years"] for r in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["emi"] > rows[-1]["emi"]


def test_thirty_percent_note_shows_thirty_percent_amount():
    q = quote(6_099_000, 1598, years=3, down_pct=20)
    assert any("30% (Rs 1,829,700)" in n for n in q["notes"])
