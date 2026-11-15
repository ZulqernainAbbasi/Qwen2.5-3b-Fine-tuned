"""The six agent tools. Every tool returns a plain string the LLM can read."""
import re
from contextvars import ContextVar
from datetime import date
from typing import Any

from langchain_core.tools import tool

import config
from agent.parsing import match_model, parse_price
from backend import db
from data.validate import load_cars
from rag.retriever import retrieve

OTHER_BRANDS = {"tesla", "bmw", "mercedes", "audi", "lexus", "nissan", "mitsubishi", "mazda", "ford",
                "chevrolet", "volkswagen", "vw", "porsche", "byd", "chery", "haval", "proton", "peugeot",
                "dfsk", "baic", "gwm", "jetour", "deepal", "ora", "seres", "prince", "faw", "isuzu"}
BODY_WORDS = {"suv": "suv", "sedan": "sedan", "hatchback": "hatchback", "crossover": "crossover",
              "pickup": "pickup", "pick up": "pickup", "van": "mpv", "mpv": "mpv"}
FUEL_WORDS = {"hybrid": "hybrid", "electric": "electric", "ev": "electric", "diesel": "diesel"}
PHONE_RE = re.compile(r"^(\+92|0)3\d{9}$")
MAX_LIST = 8
# The user's latest message, set by the agent per turn. Small models sometimes send junk
# tool arguments (e.g. the JSON schema itself); tools then fall back to this text.
CURRENT_MESSAGE: ContextVar[str] = ContextVar("CURRENT_MESSAGE", default="")


def _text(x: Any) -> str | None:
    """Return x if it is a non-empty string (or a number), else None."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return str(x)
    return x.strip() if isinstance(x, str) and x.strip() else None


def _cars():
    return load_cars()


def _line(r) -> str:
    eng = f"{r.engine_cc} cc" if str(r.engine_cc) not in ("<NA>", "nan") else "electric motor"
    mil = f", {r.mileage_kmpl} km/l" if r.mileage_kmpl else ""
    return (f"- {r.brand} {r.model} {r.variant}: Rs {r.price_pkr:,} ({eng}, {r.fuel}, {r.transmission}, "
            f"{r.body_type}, {r.seats} seats{mil}). Source: {r.source_url}")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@tool
def search_cars(query: Any = "", brand: Any = None, max_price: Any = None, body_type: Any = None) -> str:
    """Search new cars sold in Pakistan. query: the user's question as text (e.g. 'Honda City price').
    Optional: brand, max_price (e.g. '50 lakh'), body_type (suv, sedan, hatchback, crossover, pickup, mpv).
    Use this for ANY price or spec question."""
    query = _text(query) or CURRENT_MESSAGE.get() or ""
    cars = _cars()
    brand = next((b for b in cars.brand.unique() if _text(brand) and b.lower() == _text(brand).lower()), None)
    body_type = _text(body_type)
    body_type = body_type.lower() if body_type and body_type.lower() in set(BODY_WORDS.values()) else None
    max_price = _text(max_price)
    words = _words(query)
    if words & OTHER_BRANDS:
        return config.REFUSAL
    budget = parse_price(max_price) if max_price else None
    if budget is None:
        budget = parse_price(query) if re.search(r"under|below|upto|up to|tak|budget|within|less than|max", query, re.I) else None
    hit = match_model(query)
    filtered = cars
    structured = False
    if hit:
        filtered = cars[(cars.brand == hit[0]) & (cars.model == hit[1])]
        structured = True
    else:
        b = brand or next((x for x in cars.brand.unique() if x.lower() in words), None)
        if b:
            filtered = filtered[filtered.brand.str.lower() == b.lower()]
            structured = True
        bt = (body_type or "").lower() or next((v for k, v in BODY_WORDS.items() if k in query.lower()), None)
        if bt:
            bt = BODY_WORDS.get(bt, bt)
            allowed = {"suv", "crossover"} if bt == "suv" else {bt}
            filtered = filtered[filtered.body_type.isin(allowed)]
            structured = True
        fuel = next((v for k, v in FUEL_WORDS.items() if k in words), None)
        if fuel:
            filtered = filtered[filtered.fuel == fuel]
            structured = True
    if budget:
        filtered = filtered[filtered.price_pkr <= budget]
        structured = True
    if structured and len(filtered):
        rows = filtered.sort_values("price_pkr")
        if not hit:  # many models: show the cheapest matching variant of each model
            rows = rows.drop_duplicates(["brand", "model"])
        shown = rows.head(MAX_LIST)
        out = [_line(r) for r in shown.itertuples()]
        if len(rows) > MAX_LIST:
            out.append(f"({len(rows) - MAX_LIST} more not shown; narrow the budget or type)")
        out.append(config.PRICE_DISCLAIMER)
        return "\n".join(out)
    if structured and not len(filtered):
        return "No cars in the dataset match those filters. " + config.PRICE_DISCLAIMER
    hits = retrieve(query, k=5)
    if not hits:
        return config.REFUSAL
    return "\n".join(f"- {h['text'][:400]} (Source: {h['source']})" for h in hits)


@tool
def compare_models(model_a: Any, model_b: Any) -> str:
    """Compare two car models side by side (price range, engine, fuel, seats, mileage).
    model_a / model_b: model names as text, e.g. 'Toyota Corolla', 'Honda Civic'."""
    a, b = match_model(_text(model_a)), match_model(_text(model_b))
    if not a or not b:
        return config.REFUSAL
    cars = _cars()

    def summary(bm):
        rows = cars[(cars.brand == bm[0]) & (cars.model == bm[1])]
        eng = sorted({str(e) for e in rows.engine_cc if str(e) not in ("<NA>", "nan")})
        return {
            "Model": f"{bm[0]} {bm[1]}",
            "Price": f"Rs {rows.price_pkr.min():,} - Rs {rows.price_pkr.max():,}",
            "Variants": str(len(rows)),
            "Engine": ", ".join(e + " cc" for e in eng) or "electric",
            "Fuel": ", ".join(sorted(set(rows.fuel))),
            "Transmission": ", ".join(sorted(set(rows.transmission))),
            "Body": ", ".join(sorted(set(rows.body_type))),
            "Seats": ", ".join(str(s) for s in sorted(set(rows.seats))),
            "Mileage (km/l)": ", ".join(sorted({m for m in rows.mileage_kmpl if m})) or "not listed",
        }

    sa, sb = summary(a), summary(b)
    lines = [f"| | {sa['Model']} | {sb['Model']} |", "|---|---|---|"]
    lines += [f"| {k} | {sa[k]} | {sb[k]} |" for k in sa if k != "Model"]
    return "\n".join(lines) + "\n" + config.PRICE_DISCLAIMER


@tool
def emi_calculator(price: Any, down_payment_pct: float = 30, years: int = 5, annual_rate_pct: float = 20) -> str:
    """Calculate a monthly car-loan instalment (EMI). price can be '50 lakh', '1.2 crore' or 'Rs 6,099,000'."""
    p = parse_price(_text(price))
    if not p:
        return "Cannot calculate: I couldn't understand the price. Give it like '50 lakh' or 'Rs 6,099,000'."
    if not (0 <= down_payment_pct < 100) or not (1 <= years <= 10) or annual_rate_pct < 0:
        return "Cannot calculate: down payment must be 0-99%, tenure 1-10 years, rate >= 0."
    down = round(p * down_payment_pct / 100)
    loan = p - down
    n = int(years * 12)
    r = annual_rate_pct / 100 / 12
    emi = loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total_interest = emi * n - loan
    return (f"Price: Rs {p:,}. Down payment ({down_payment_pct:g}%): Rs {down:,}. Loan: Rs {loan:,}. "
            f"Monthly instalment for {years} years at {annual_rate_pct:g}% per year: Rs {round(emi):,}. "
            f"Total markup paid: Rs {round(total_interest):,}. (Estimate; banks' actual rates vary.)")


def _validate_booking(name: str, phone: str, model: str, when: str):
    clean = re.sub(r"[\s-]", "", phone or "")
    if not name or not name.strip():
        return None, "Cannot book: name is required."
    if not PHONE_RE.match(clean):
        return None, "Cannot book: phone must be a Pakistani mobile number like 03001234567."
    hit = match_model(model)
    if not hit:
        return None, f"Cannot book: model '{model}' is not one of the cars we sell."
    try:
        d = date.fromisoformat(when)
    except (TypeError, ValueError):
        return None, "Cannot book: date must be in YYYY-MM-DD format."
    if d < date.today():
        return None, "Cannot book: date is in the past."
    norm_phone = "+92" + clean[1:] if clean.startswith("0") else clean
    return {"name": name.strip(), "phone": norm_phone, "model": f"{hit[0]} {hit[1]}", "date": d.isoformat()}, None


@tool
def book_test_drive(name: str, phone: str, model: str, city: str, date: str) -> str:
    """Book a test drive. Only call AFTER the customer confirmed all details. date = YYYY-MM-DD."""
    fields, err = _validate_booking(name, phone, model, date)
    if err:
        return err
    if not city or not city.strip():
        return "Cannot book: city is required."
    bid = db.add_booking("test_drive", city=city.strip(), **fields)
    return f"Booked #{bid}: test drive of {fields['model']} for {fields['name']} in {city.strip()} on {fields['date']}. The dealer will call {fields['phone']} to confirm."


@tool
def book_service(name: str, phone: str, model: str, km: int, date: str) -> str:
    """Book a periodic service appointment. Only call AFTER the customer confirmed all details."""
    fields, err = _validate_booking(name, phone, model, date)
    if err:
        return err
    if km is None or int(km) < 0:
        return "Cannot book: odometer km must be a positive number."
    bid = db.add_booking("service", km=int(km), **fields)
    return f"Booked #{bid}: service for {fields['model']} ({int(km):,} km) on {fields['date']} for {fields['name']}."


@tool
def escalate_to_human(summary: str) -> str:
    """Hand the conversation to a human staff member (complaints, negotiations, anything you cannot do)."""
    tid = db.add_ticket(summary)
    return f"Ticket #{tid} created; a staff member will contact you."


ALL_TOOLS = [search_cars, compare_models, emi_calculator, book_test_drive, book_service, escalate_to_human]
