"""Car financing in Pakistan: State Bank (SBP) rules + indicative bank quotes. Pure Python (used by agent and UI).

SBP prudential regulations (in force, Nov 2026):
- minimum down payment 30% (SBP BPRD circular 2021),
- total auto financing per person capped at Rs 3,000,000 (same circular),
- maximum tenure 5 years for cars up to 1,000cc and 3 years above 1,000cc (SBP, 2022).
A draft policy (AIDEP 2026-31) proposes 7 years / 15% down / Rs 10m cap, but it is not law yet.
"""
import csv
from pathlib import Path

BANKS_CSV = Path(__file__).resolve().parents[1] / "data" / "banks.csv"
MIN_DOWN_PCT = 30
MAX_LOAN = 3_000_000
KIBOR_1Y = 12.19  # SBP KIBOR 1-year offer, 4 Sep 2026
SBP_NOTE = ("SBP rules: minimum 30% down payment, total car financing capped at Rs 3,000,000 per person, "
            "maximum 5 years for cars up to 1,000cc and 3 years above 1,000cc.")


def emi(loan: float, rate_pct: float, years: int) -> float:
    n = int(years * 12)
    r = rate_pct / 100 / 12
    return loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)


def sbp_terms(price: int, engine_cc: int | None) -> dict:
    small = engine_cc is not None and 0 < int(engine_cc) <= 1000
    return {"min_down_pct": MIN_DOWN_PCT, "max_loan": MAX_LOAN, "max_years": 5 if small else 3,
            "min_down": max(round(price * MIN_DOWN_PCT / 100), price - MAX_LOAN)}


def load_banks(path: Path = BANKS_CSV) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{**r, "rate_low_pct": float(r["rate_low_pct"]), "rate_high_pct": float(r["rate_high_pct"])}
                for r in csv.DictReader(f)]


def quote(price: int, engine_cc: int | None, years: int = 3, down_pct: float = 30) -> dict:
    """Apply SBP rules, then compute each bank's monthly instalment range for the allowed loan."""
    if years < 1 or price <= 0:
        raise ValueError("price must be positive and years at least 1")
    t = sbp_terms(price, engine_cc)
    notes = []
    down = round(price * down_pct / 100)
    if down_pct < MIN_DOWN_PCT:
        notes.append(f"Down payment raised to the SBP minimum of 30% (Rs {round(price * MIN_DOWN_PCT / 100):,}).")
    if down < t["min_down"]:
        if price - down > MAX_LOAN:
            notes.append(f"Banks can finance at most Rs 3,000,000 per person (SBP cap), so the down payment must be "
                         f"at least Rs {t['min_down']:,}.")
        down = t["min_down"]
    if years > t["max_years"]:
        notes.append(f"SBP allows at most {t['max_years']} years for this car "
                     f"({'up to' if t['max_years'] == 5 else 'above'} 1,000cc), so the tenure is set to "
                     f"{t['max_years']} years.")
        years = t["max_years"]
    loan = price - down
    rows = []
    for b in load_banks():
        lo, hi = emi(loan, b["rate_low_pct"], years), emi(loan, b["rate_high_pct"], years)
        rows.append({"bank": b["bank"], "product": b["product"], "type": b["type"],
                     "rate_low": b["rate_low_pct"], "rate_high": b["rate_high_pct"], "basis": b["rate_basis"],
                     "source": b["source_url"], "emi_low": round(lo), "emi_high": round(hi),
                     "total_low": round(round(lo) * years * 12), "total_high": round(round(hi) * years * 12)})
    rows.sort(key=lambda r: r["emi_low"])
    return {"price": price, "down": down, "loan": loan, "years": years, "max_years": t["max_years"],
            "notes": notes, "banks": rows}


def emi_by_years(price: int, engine_cc: int | None, down_pct: float = 30, rate_pct: float = 15) -> list[dict]:
    """Monthly instalment for every tenure SBP allows for this car, at one rate."""
    t = sbp_terms(price, engine_cc)
    down = max(round(price * max(down_pct, MIN_DOWN_PCT) / 100), t["min_down"])
    loan = price - down
    return [{"years": y, "emi": round(emi(loan, rate_pct, y)), "total": round(emi(loan, rate_pct, y) * y * 12),
             "markup": round(emi(loan, rate_pct, y) * y * 12 - loan)} for y in range(1, t["max_years"] + 1)]
