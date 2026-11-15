"""Three-way evaluation (base / base+RAG / fine-tuned+RAG) plus an agent tool-use check.

Run (Windows, Ollama running):
  python -m eval.run_eval --config base
  python -m eval.run_eval --config base_rag
  python -m eval.run_eval --config ft_rag
  python -m eval.run_eval --agent            # tool-call success rate of the full agent
  python -m eval.run_eval --summary          # table + chart from results.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import requests

import config
from eval.score import score
from finetune.build_dataset import SYSTEM_PROMPT

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
FIELDS = ["config", "id", "kind", "latency_s", "price_correct", "refusal_correct", "keyword_recall", "answer"]
CONFIGS = {"base": ("qwen2.5:3b-instruct", False), "base_rag": ("qwen2.5:3b-instruct", True),
           "ft_rag": ("car-assistant", True)}
QA_KINDS = {"spec", "recommendation", "comparison", "refusal", "roman_urdu", "handwritten"}


def context_for(question: str) -> str:
    from agent.tools import search_cars
    return search_cars.invoke({"query": question})


def ask(model: str, question: str, ctx: str | None) -> str:
    user = question if ctx is None else f"Context from the car database:\n{ctx}\n\nQuestion: {question}"
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=180, json={
        "model": model, "stream": False, "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def run_config(name: str, limit: int | None) -> None:
    model, use_rag = CONFIGS[name]
    items = [json.loads(l) for l in (config.DATA_DIR / "finetune" / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    items = [e for e in items if e["type"] in QA_KINDS][:limit]
    new = not RESULTS.exists()
    done = set() if new else {r["id"] for r in csv.DictReader(open(RESULTS, encoding="utf-8")) if r["config"] == name}
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, e in enumerate(items):
            if str(i) in done:  # resume an interrupted run
                continue
            q = e["messages"][1]["content"]
            expected = e["messages"][2]["content"]
            t0 = time.time()
            try:
                ans = ask(model, q, context_for(q) if use_rag else None)
            except requests.RequestException as err:  # e.g. a runaway generation timing out
                ans = f"[error: {type(err).__name__}]"
            s = score(ans, expected, e["type"])
            w.writerow({"config": name, "id": i, "kind": e["type"], "latency_s": round(time.time() - t0, 2),
                        **s, "answer": ans[:500]})
            print(f"[{name}] {i + 1}/{len(items)} {e['type']}: {s}", flush=True)


AGENT_TASKS = [
    ("What is the price of the Suzuki Alto?", "search_cars"),
    ("Compare Toyota Corolla and Honda Civic", "compare_models"),
    ("Monthly instalment for a 60 lakh car with 30% down for 5 years?", "emi_calculator"),
    ("Best SUV under 1 crore?", "search_cars"),
    ("Hyundai Tucson vs KIA Sportage", "compare_models"),
    ("80 lakh ki gaari ki 3 saal ki qist kitni hogi?", "emi_calculator"),
    ("I want to complain, the dealer overcharged me. Let me talk to a human.", "escalate_to_human"),
    ("Honda City price?", "search_cars"),
    ("Yes confirmed: book a test drive for Ali, 03001234567, Toyota Corolla, Lahore, 2027-01-10", "book_test_drive"),
    ("Yes confirmed: book service for Sara, 03211234567, Honda City, 10000 km, 2027-01-12", "book_service"),
]


def run_agent_eval() -> None:
    config.DB_PATH = HERE.parent / "outputs" / "eval_bookings.db"  # keep demo bookings clean
    from agent.graph import run_agent
    ok = 0
    rows = []
    for i, (msg, tool) in enumerate(AGENT_TASKS):
        t0 = time.time()
        out = run_agent(f"eval-{i}-{time.time()}", msg)
        used = [s["tool"] for s in out["steps"]]
        errors = [s for s in out["steps"] if "Error invoking tool" in s["result"] or s["result"].startswith("Cannot")]
        hit = tool in used and not any(e["tool"] == tool for e in errors)
        ok += hit
        rows.append({"task": msg, "expected_tool": tool, "used": "|".join(used), "success": hit,
                     "latency_s": round(time.time() - t0, 1)})
        print(f"{'OK ' if hit else 'MISS'} {tool:18s} used={used}")
    with open(HERE / "agent_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"tool-call success rate: {ok}/{len(AGENT_TASKS)}")


def summary() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(RESULTS, encoding="utf-8")))
    names = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    metrics = {}
    for c in names:
        rs = [r for r in rows if r["config"] == c]
        mean = lambda key: (lambda v: sum(v) / len(v) if v else 0.0)([r[key] == "True" for r in rs if r[key] in ("True", "False")])
        metrics[c] = {"price accuracy": mean("price_correct"), "refusal accuracy": mean("refusal_correct"),
                      "keyword recall": sum(float(r["keyword_recall"]) for r in rs) / len(rs),
                      "avg latency (s)": sum(float(r["latency_s"]) for r in rs) / len(rs)}
    print(f"{'config':10s} " + " ".join(f"{k:>18s}" for k in next(iter(metrics.values()))))
    for c, m in metrics.items():
        print(f"{c:10s} " + " ".join(f"{v:18.3f}" for v in m.values()))
    (HERE / "charts").mkdir(exist_ok=True)
    keys = ["price accuracy", "refusal accuracy", "keyword recall"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(names)
    for i, c in enumerate(names):
        ax.bar([k + i * width for k in range(len(keys))], [metrics[c][k] for k in keys], width, label=c)
    ax.set_xticks([k + width * (len(names) - 1) / 2 for k in range(len(keys))], keys)
    ax.set_ylim(0, 1)
    ax.set_title("PakCar Assistant: base vs RAG vs fine-tuned + RAG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "charts" / "summary.png", dpi=150)
    print("chart:", HERE / "charts" / "summary.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.config:
        run_config(a.config, a.limit)
    if a.agent:
        run_agent_eval()
    if a.summary:
        summary()
