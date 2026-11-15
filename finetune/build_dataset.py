"""Build chat-format fine-tuning data (train/eval JSONL) from cars.csv.

Every price an assistant turn mentions comes straight from cars.csv, so the model
learns the advisor style without learning invented numbers.

Run: python -m finetune.build_dataset
"""
import json
import random
from pathlib import Path

import config
from data.validate import load_cars

SYSTEM_PROMPT = (
    "You are PakCar Assistant, a friendly and honest sales and service advisor for new cars sold in "
    "Pakistan (Toyota, Honda, Suzuki, KIA, Hyundai, Changan, MG). Answer in the user's language "
    "(English or Roman Urdu). Only state prices and specs you were given or found with a tool; if you "
    f"do not have verified information, say exactly: \"{config.REFUSAL}\" and offer what you do know. "
    "Prices are ex-factory as of Nov 2026."
)
TOOL_NAMES = ["search_cars", "compare_models", "emi_calculator", "book_test_drive", "book_service",
              "escalate_to_human"]
MIN_COUNTS = {"spec": 300, "recommendation": 200, "comparison": 150, "refusal": 120,
              "roman_urdu": 100, "tool_call": 100, "handwritten": 40}
HANDWRITTEN = Path(__file__).with_name("handwritten.jsonl")
DISCLAIMER = "Prices are ex-factory (Nov 2026); please confirm with an authorised dealer."


def rs(n: int) -> str:
    return f"Rs {int(n):,}"


def lakh(n: int) -> str:
    return f"{n / 100_000:.1f}".rstrip("0").rstrip(".") + " lakh"


def ex(kind: str, user: str, assistant: str) -> dict:
    return {"type": kind, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": user},
                                       {"role": "assistant", "content": assistant}]}


def _eng(r) -> str:
    return f"{r.engine_cc} cc" if str(r.engine_cc) not in ("<NA>", "nan") else "an electric motor"


def gen_spec(cars, rng):
    qs = ["What is the price of the {n}?", "How much does the {n} cost?", "{n} price in Pakistan?",
          "Tell me about the {n}.", "What engine does the {n} have?", "Is the {n} manual or automatic?",
          "How many seats does the {n} have?", "Give me the specs of {n}.", "{n} ka rate kya hai? (English please)",
          "What's the ex-factory price of {n}?"]
    out = []
    for r in cars.itertuples():
        name = f"{r.brand} {r.model} {r.variant}"
        mil = f" Its fuel average is about {r.mileage_kmpl} km/l." if r.mileage_kmpl else ""
        feats = f" Key features: {r.key_features.replace(';', ', ')}." if r.key_features else ""
        ans = (f"The {name} costs {rs(r.price_pkr)} ex-factory. It has {_eng(r)}, runs on {r.fuel}, "
               f"comes with a {r.transmission} transmission and seats {r.seats}.{mil}{feats} {DISCLAIMER}")
        for q in rng.sample(qs, 5):
            out.append(ex("spec", q.format(n=name), ans))
    # model-level range questions
    for (b, m), g in cars.groupby(["brand", "model"]):
        lo, hi = g.price_pkr.min(), g.price_pkr.max()
        variants = ", ".join(f"{v} ({rs(p)})" for v, p in zip(g.variant, g.price_pkr))
        ans = f"The {b} {m} is available in {len(g)} variant(s): {variants}. {DISCLAIMER}"
        for q in ["What variants does the {n} come in?", "{n} price range?", "List all {n} variants with prices."]:
            out.append(ex("spec", q.format(n=f"{b} {m}"), ans))
    return out


def gen_recommendation(cars, rng):
    budgets = [3_000_000, 3_500_000, 4_000_000, 4_500_000, 5_000_000, 5_500_000, 6_000_000, 6_500_000,
               7_000_000, 7_500_000, 8_000_000, 9_000_000, 10_000_000, 12_000_000, 15_000_000]
    needs = {
        "": None, "for a small family": None, "for city driving": {"hatchback", "sedan", "crossover"},
        "SUV": {"suv", "crossover"}, "sedan": {"sedan"}, "7 seater": "seats7", "hybrid": "hybrid",
        "automatic": "auto", "electric": "electric", "hatchback": {"hatchback"},
    }
    templates = ["Best {need} car under {b}?", "My budget is {b}. Which {need} car should I buy?",
                 "Suggest a {need} car within {b}.", "What {need} cars can I get for {b}?"]
    out = []
    for budget in budgets:
        for need, rule in needs.items():
            pool = cars[cars.price_pkr <= budget]
            if isinstance(rule, set):
                pool = pool[pool.body_type.isin(rule)]
            elif rule == "seats7":
                pool = pool[pool.seats >= 7]
            elif rule in ("hybrid", "electric"):
                pool = pool[pool.fuel == rule]
            elif rule == "auto":
                pool = pool[pool.transmission == "automatic"]
            b = lakh(budget) if budget < 10_000_000 else f"{budget / 10_000_000:g} crore"
            q = rng.choice(templates).format(need=need, b=b).replace("  ", " ")
            if pool.empty:
                ans = (f"None of the new cars in my data fit {need + ' ' if need else ''}under {b}. "
                       "Would you like me to look at a slightly higher budget?")
            else:
                picks = pool.sort_values("price_pkr", ascending=False).drop_duplicates(["brand", "model"]).head(3)
                lines = [f"- {r.brand} {r.model} {r.variant}: {rs(r.price_pkr)} ({r.fuel}, {r.transmission}, "
                         f"{r.body_type}, {r.seats} seats)" for r in picks.itertuples()]
                ans = (f"Here are good {need + ' ' if need else ''}options under {b}:\n" + "\n".join(lines) +
                       f"\nThe first one gives you the most car for your budget. {DISCLAIMER} "
                       "Shall I compare two of these or work out a monthly instalment?")
            out.append(ex("recommendation", q, ans))
            out.append(ex("recommendation", rng.choice(templates).format(need=need, b=b).replace("  ", " ") + " Thanks", ans))
    return out


def gen_comparison(cars, rng):
    models = list(cars.groupby(["brand", "model"]))
    out, seen = [], set()
    templates = ["Compare {a} and {b}.", "{a} vs {b}, which is better?", "Should I buy the {a} or the {b}?",
                 "Difference between {a} and {b}?"]
    pairs = [(i, j) for i in range(len(models)) for j in range(len(models)) if i < j]
    rng.shuffle(pairs)
    for i, j in pairs[:170]:
        (ka, ga), (kb, gb) = models[i], models[j]
        a, b = " ".join(ka), " ".join(kb)
        desc = lambda k, g: (f"{' '.join(k)}: {rs(g.price_pkr.min())} to {rs(g.price_pkr.max())}, "
                             f"{', '.join(sorted(set(g.fuel)))}, {', '.join(sorted(set(g.body_type)))}, "
                             f"{max(g.seats)} seats")
        cheaper = a if ga.price_pkr.min() <= gb.price_pkr.min() else b
        ans = (f"- {desc(ka, ga)}\n- {desc(kb, gb)}\nThe {cheaper} is the cheaper way in. The better choice depends "
               f"on your budget, seating needs and fuel preference. {DISCLAIMER}")
        q = rng.choice(templates).format(a=a, b=b)
        if q not in seen:
            seen.add(q)
            out.append(ex("comparison", q, ans))
    return out


def gen_refusal(cars, rng):
    unknown = ["Tesla Model 3", "BMW X5", "Mercedes C200", "Audi A4", "Toyota Land Cruiser 300", "Honda Accord",
               "Haval H6", "BYD Atto 3", "Chery Tiggo 8", "Proton Saga", "Nissan Sunny", "Mazda CX-5",
               "Hyundai Ioniq 5", "Kia Carnival", "Suzuki Jimny", "Peugeot 2008", "DFSK Glory", "Toyota Camry",
               "Toyota Prius", "Suzuki WagonR"]
    q_unknown = ["What is the price of the {n}?", "Is the {n} available in Pakistan?", "{n} specs please",
                 "{n} ki price batao"]
    out = []
    for n in unknown:
        for q in rng.sample(q_unknown, 3):
            out.append(ex("refusal", q.format(n=n),
                          f"{config.REFUSAL} The {n} isn't in my current Pakistan new-car data. I can help with "
                          "Toyota, Honda, Suzuki, KIA, Hyundai, Changan and MG models. Want some similar options?"))
    unknown_specs = ["top speed", "boot space", "warranty period", "0-100 km/h time", "ground clearance",
                     "fuel tank size", "service cost", "resale value in 3 years"]
    for (b, m), g in cars.groupby(["brand", "model"]):
        for spec in rng.sample(unknown_specs, 2):
            out.append(ex("refusal", f"What is the {spec} of the {b} {m}?",
                          f"{config.REFUSAL} For the {b} {m} I can tell you the price ({rs(g.price_pkr.min())} "
                          f"onwards), engine, fuel type, transmission and seats. Would that help?"))
    return out


def gen_roman_urdu(cars, rng):
    out = []
    qs = ["{n} ki qeemat kya hai?", "{n} kitne ki hai?", "{n} ka price batayein", "{n} ke baare mein batao"]
    for (b, m), g in cars.groupby(["brand", "model"]):
        cheapest = g.sort_values("price_pkr").iloc[0]
        ans = (f"{b} {m} ki ex-factory qeemat {rs(g.price_pkr.min())} se shuru hoti hai "
               f"({cheapest.variant}) aur top variant {rs(g.price_pkr.max())} ka hai. Yeh {', '.join(sorted(set(g.fuel)))} "
               f"gaari hai jismein {max(g.seats)} seats hain. Qeematein Nov 2026 ki hain, dealer se confirm zaroor karein.")
        for q in qs:
            out.append(ex("roman_urdu", q.format(n=f"{b} {m}"), ans))
    for budget in [3_000_000, 4_000_000, 5_000_000, 6_000_000, 8_000_000, 10_000_000]:
        pool = cars[cars.price_pkr <= budget].sort_values("price_pkr", ascending=False).drop_duplicates(["brand", "model"]).head(3)
        lines = "\n".join(f"- {r.brand} {r.model} {r.variant}: {rs(r.price_pkr)}" for r in pool.itertuples())
        for q in [f"{lakh(budget)} mein konsi gaari achi hai?", f"Mera budget {lakh(budget)} hai, koi gaari batao",
                  f"{lakh(budget)} tak family car chahiye"]:
            out.append(ex("roman_urdu", q, f"{lakh(budget)} ke andar yeh behtareen options hain:\n{lines}\n"
                                            "Kya main inka comparison ya qist (EMI) nikaal doon?"))
    return out


def _call(tool_name: str, **args) -> str:
    return "<tool_call>\n" + json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def gen_tool_call(cars, rng):
    out = []
    models = [f"{b} {m}" for b, m in cars[["brand", "model"]].drop_duplicates().itertuples(index=False)]
    for m in rng.sample(models, 25):
        out.append(ex("tool_call", f"Latest price of {m}?", _call("search_cars", query=f"{m} price")))
    for a, b in [rng.sample(models, 2) for _ in range(20)]:
        out.append(ex("tool_call", f"Can you compare {a} with {b} for me?", _call("compare_models", model_a=a, model_b=b)))
    for p, d, y in [(rng.choice([30, 40, 50, 60, 70, 80, 90]), rng.choice([20, 30, 40, 50]), rng.choice([3, 5, 7])) for _ in range(35)]:
        out.append(ex("tool_call", f"Monthly installment for a {p} lakh car with {d}% down payment for {y} years?",
                      _call("emi_calculator", price=f"{p} lakh", down_payment_pct=d, years=y)))
    names = ["Ali", "Sara", "Ahmed", "Fatima", "Usman", "Ayesha", "Bilal", "Hina", "Zain", "Maryam"]
    cities = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar"]
    for i in range(20):
        n, c, m = rng.choice(names), rng.choice(cities), rng.choice(models)
        ph = f"03{rng.randint(0, 4)}{rng.randint(10_000_000, 99_999_999)}"
        dt = f"2026-12-{rng.randint(1, 28):02d}"
        out.append(ex("tool_call", f"Yes, confirmed. Book my test drive: {n}, {ph}, {m}, {c}, {dt}.",
                      _call("book_test_drive", name=n, phone=ph, model=m, city=c, date=dt)))
    for i in range(10):
        n, m = rng.choice(names), rng.choice(models)
        ph = f"030{rng.randint(10_000_000, 99_999_999)}"
        km = rng.choice([1000, 5000, 10000, 20000, 40000])
        dt = f"2026-12-{rng.randint(1, 28):02d}"
        out.append(ex("tool_call", f"Confirmed, please book service for my {m}, {km} km, name {n}, phone {ph}, on {dt}.",
                      _call("book_service", name=n, phone=ph, model=m, km=km, date=dt)))
    for s in ["My car was delivered with a scratch and the dealer is not responding",
              "I want a bigger discount on a Fortuner, can I talk to a manager?",
              "I need to cancel my booking and get a refund", "Mujhe kisi insaan se baat karni hai",
              "Can someone call me about fleet pricing for 10 cars?", "The service center overcharged me"]:
        out.append(ex("tool_call", s, _call("escalate_to_human", summary=s)))
    return out


def load_handwritten() -> list[dict]:
    rows = []
    for line in HANDWRITTEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.append(ex("handwritten", d["user"], d["assistant"]))
    return rows


def build(seed: int = 42, out_dir: Path | None = None) -> tuple[int, int]:
    rng = random.Random(seed)
    cars = load_cars()
    out_dir = Path(out_dir or config.DATA_DIR / "finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [gen_spec(cars, rng), gen_recommendation(cars, rng), gen_comparison(cars, rng),
              gen_refusal(cars, rng), gen_roman_urdu(cars, rng), gen_tool_call(cars, rng), load_handwritten()]
    seen, by_type = set(), {}
    for g in groups:
        for e in g:
            key = " ".join(e["messages"][1]["content"].lower().split())
            if key in seen:
                continue
            seen.add(key)
            by_type.setdefault(e["type"], []).append(e)
    train, evals = [], []
    for kind, items in by_type.items():
        rng.shuffle(items)
        k = max(1, round(len(items) * 0.10))
        evals += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(evals)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(train), len(evals)


if __name__ == "__main__":
    n_train, n_eval = build()
    print(f"train={n_train} eval={n_eval} -> {config.DATA_DIR / 'finetune'}")
