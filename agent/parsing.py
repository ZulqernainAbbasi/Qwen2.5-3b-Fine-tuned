"""Parse Pakistani price expressions and fuzzy-match car model names."""
import re
from functools import lru_cache

from rapidfuzz import fuzz

UNITS = {
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "lak": 100_000,
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000, "karor": 10_000_000,
    "million": 1_000_000, "mn": 1_000_000, "m": 1_000_000,
}
_PRICE_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|lak|crores?|cr|karor|million|mn|m)?(?![a-z])", re.I)


def parse_price(text: str | None) -> int | None:
    """Return the first price in `text` as whole PKR, or None."""
    if not text:
        return None
    for num, unit in _PRICE_RE.findall(str(text)):
        value = float(num.replace(",", ""))
        if unit:
            return int(round(value * UNITS[unit.lower()]))
        if value >= 100_000:  # bare number: only plausible as rupees if large
            return int(value)
    return None


@lru_cache(maxsize=1)
def _catalog() -> tuple[tuple[str, str], ...]:
    from data.validate import load_cars
    cars = load_cars()
    return tuple(cars[["brand", "model"]].drop_duplicates().itertuples(index=False, name=None))


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def match_model(name: str | None) -> tuple[str, str] | None:
    """Map free text (typos, Roman Urdu) to a (brand, model) in the dataset, or None."""
    if not name:
        return None
    q = _norm(name)
    padded = f" {q} "
    # 1) exact whole-word containment; prefer the longest model name ("corolla cross" over "corolla")
    exact = [(b, m) for b, m in _catalog() if f" {_norm(m)} " in padded or f" {_norm(m).replace(' ', '')} " in padded]
    if exact:
        return max(exact, key=lambda bm: len(bm[1]))
    # 2) fuzzy match of 1-3 word windows against model names (typos like "corola")
    words = q.split()
    grams = {" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    best, best_score = None, 0.0
    for b, m in _catalog():
        target = _norm(m)
        if len(target.replace(" ", "")) <= 3:  # short names (HS, ZS, MG4) need exact match
            continue
        for g in grams:
            s = fuzz.ratio(g, target)
            if s > best_score or (s == best_score and best and len(m) > len(best[1])):
                best, best_score = (b, m), s
    return best if best_score >= 80 else None
