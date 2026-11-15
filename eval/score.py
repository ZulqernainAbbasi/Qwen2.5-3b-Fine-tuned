"""Score a model answer against the reference answer from eval.jsonl."""
import re

from agent.parsing import UNITS
from config import REFUSAL

_RS = re.compile(r"Rs\.?\s?([\d,]{5,})", re.I)
_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|million)\b", re.I)
_STOP = {"the", "a", "an", "is", "it", "and", "or", "of", "in", "on", "for", "with", "to", "rs", "pkr",
         "ex", "factory", "nov", "2026", "please", "confirm", "dealer", "prices", "are", "authorised",
         "an", "has", "its", "you", "your", "i", "can", "my", "me", "this", "that", "be", "at", "as"}


def _prices(text: str) -> list[int]:
    vals = [int(n.replace(",", "")) for n in _RS.findall(text)]
    vals += [int(round(float(n) * UNITS[u.lower().rstrip("s")])) for n, u in _UNIT.findall(text)]
    return vals


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def score(answer: str, expected: str, kind: str) -> dict:
    """price_correct: every price in `expected` appears in `answer` (±0.5%), None if expected has none.
    refusal_correct: for refusal items, answer contains the refusal and states no price.
    keyword_recall: share of expected content words present in the answer."""
    exp_prices = _prices(expected)
    got = _prices(answer)
    price_ok = None
    if exp_prices and kind != "refusal":
        price_ok = all(any(abs(g - p) <= 0.005 * p for g in got) for p in exp_prices)
    refusal_ok = None
    if kind == "refusal":
        refusal_ok = REFUSAL.lower().rstrip(".") in answer.lower() and not any(
            p >= 1_000_000 for p in got if p not in exp_prices)
    exp_words = _words(expected)
    recall = len(exp_words & _words(answer)) / len(exp_words) if exp_words else 1.0
    return {"price_correct": price_ok, "refusal_correct": refusal_ok, "keyword_recall": round(recall, 3)}
