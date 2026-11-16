"""Car catalogue helpers for the Streamlit app (no heavy dependencies)."""
import csv
import re
from pathlib import Path

COMMON_WORDS = {"city", "every", "swift", "cross"}


def fmt_pkr(n: int) -> str:
    """Pakistani short form: 46.5 lakh, 1.05 crore."""
    if n >= 10_000_000:
        return f"{n / 10_000_000:.2f}".rstrip("0").rstrip(".") + " crore"
    return f"{n / 100_000:.1f}".rstrip("0").rstrip(".") + " lakh"


def load_models(root: Path) -> list[dict]:
    """One dict per model with its variants, price range and (optional) image credit."""
    root = Path(root)
    images = {}
    img_path = root / "data" / "images.csv"
    if img_path.exists():
        with open(img_path, encoding="utf-8") as f:
            images = {(r["brand"], r["model"]): r for r in csv.DictReader(f)}
    models: dict = {}
    with open(root / "data" / "cars.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["brand"], r["model"])
            m = models.setdefault(key, {"brand": r["brand"], "model": r["model"], "variants": [],
                                        "body_types": set(), "fuels": set(), "seats": set(), "features": set()})
            price = int(r["price_pkr"])
            m["variants"].append({"variant": r["variant"], "price": price, "engine_cc": r["engine_cc"],
                                  "fuel": r["fuel"], "transmission": r["transmission"],
                                  "mileage": r["mileage_kmpl"], "source": r["source_url"]})
            m["body_types"].add(r["body_type"])
            m["fuels"].add(r["fuel"])
            m["seats"].add(int(r["seats"]))
            m["features"].update(x for x in r["key_features"].split(";") if x)
    out = []
    for key, m in models.items():
        m["variants"].sort(key=lambda v: v["price"])
        m["min_price"], m["max_price"] = m["variants"][0]["price"], m["variants"][-1]["price"]
        m["image"] = images.get(key)
        out.append(m)
    return out


def _patterns(m: dict) -> list[re.Pattern]:
    name = re.escape(m["model"])
    pats = [re.compile(rf"\b{re.escape(m['brand'])}\s+{name}\b", re.I)]
    if len(m["model"]) > 3 and m["model"].lower() not in COMMON_WORDS:
        pats.append(re.compile(rf"\b{name}\b", re.I))
    return pats


def models_in_text(text: str, models: list[dict]) -> list[dict]:
    """Models mentioned in text, in order of first mention; longest names win overlaps."""
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, dict]] = []
    for m in sorted(models, key=lambda m: len(m["brand"] + m["model"]), reverse=True):
        for pat in _patterns(m):
            hit = next((x for x in pat.finditer(text or "")
                        if not any(x.start() < e and x.end() > s for s, e in claimed)), None)
            if hit:
                claimed.append((hit.start(), hit.end()))
                found.append((hit.start(), m))
                break
    return [m for _, m in sorted(found, key=lambda t: t[0])]
