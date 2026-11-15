import json
import re
from collections import Counter

import pytest

import config
from data.validate import load_cars
from finetune.build_dataset import MIN_COUNTS, SYSTEM_PROMPT, TOOL_NAMES, build

PRICE_RE = re.compile(r"Rs\.? ?([\d,]{7,})")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("ft")
    n_train, n_eval = build(seed=42, out_dir=out)
    train = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [json.loads(l) for l in (out / "eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (n_train, n_eval) == (len(train), len(evals))
    return train, evals


def _user(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "user")


def _assistant(ex):
    return [m["content"] for m in ex["messages"] if m["role"] == "assistant"]


def test_counts_meet_minimums(dataset):
    train, evals = dataset
    counts = Counter(ex["type"] for ex in train + evals)
    for kind, minimum in MIN_COUNTS.items():
        assert counts[kind] >= minimum, (kind, counts[kind])


def test_split_is_about_90_10_and_stratified(dataset):
    train, evals = dataset
    assert 0.08 <= len(evals) / (len(train) + len(evals)) <= 0.12
    assert set(ex["type"] for ex in evals) == set(MIN_COUNTS)


def test_chat_format(dataset):
    train, _ = dataset
    for ex in train[:50]:
        roles = [m["role"] for m in ex["messages"]]
        assert roles[0] == "system" and roles[1] == "user" and "assistant" in roles
        assert ex["messages"][0]["content"].startswith(SYSTEM_PROMPT[:40])


def test_every_price_in_answers_exists_in_csv(dataset):
    train, evals = dataset
    # allowed: CSV prices + figures stated in the sourced docs (taxes, freight)
    prices = set(load_cars().price_pkr)
    for doc in config.DOCS_DIR.glob("*.md"):
        prices |= {int(n.replace(",", "")) for n in re.findall(r"Rs ([\d,]+)", doc.read_text(encoding="utf-8"))}
    for ex in train + evals:
        for text in _assistant(ex):
            for num in PRICE_RE.findall(text):
                assert int(num.replace(",", "")) in prices, (num, text[:120])


def test_no_eval_train_overlap(dataset):
    train, evals = dataset
    norm = lambda s: " ".join(s.lower().split())
    assert not {norm(_user(e)) for e in train} & {norm(_user(e)) for e in evals}


def test_refusals_use_exact_string(dataset):
    train, evals = dataset
    for ex in train + evals:
        if ex["type"] == "refusal":
            assert _assistant(ex)[0].startswith(config.REFUSAL)


def test_tool_call_examples_name_real_tools(dataset):
    train, evals = dataset
    for ex in train + evals:
        if ex["type"] == "tool_call":
            body = _assistant(ex)[0]
            m = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", body, re.S)
            assert m, body
            call = json.loads(m.group(1))
            assert call["name"] in TOOL_NAMES and isinstance(call["arguments"], dict)


def test_handwritten_loaded(dataset):
    train, evals = dataset
    assert sum(ex["type"] == "handwritten" for ex in train + evals) >= 40
