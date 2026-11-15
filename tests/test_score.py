from config import REFUSAL
from eval.score import score


def test_exact_price_is_correct():
    s = score("The Corolla costs Rs 6,099,000.", "The Corolla is Rs 6,099,000 ex-factory.", "spec")
    assert s["price_correct"] is True


def test_price_within_half_percent_is_correct():
    assert score("about Rs 6,120,000", "Rs 6,099,000", "spec")["price_correct"] is True


def test_wrong_price_is_incorrect():
    assert score("It costs Rs 5,500,000.", "It costs Rs 6,099,000.", "spec")["price_correct"] is False


def test_lakh_format_answer_counts():
    assert score("Price is 60.99 lakh", "Rs 6,099,000", "spec")["price_correct"] is True


def test_no_expected_price_gives_none():
    assert score("hello", "Hi! How can I help?", "handwritten")["price_correct"] is None


def test_refusal_expected_and_given():
    s = score(f"{REFUSAL} Try Toyota.", f"{REFUSAL} The Tesla is not in my data.", "refusal")
    assert s["refusal_correct"] is True


def test_refusal_expected_but_invented_price():
    s = score("The Tesla Model 3 costs Rs 15,000,000.", f"{REFUSAL} Not in my data.", "refusal")
    assert s["refusal_correct"] is False


def test_refusal_not_applicable():
    assert score("x", "Rs 6,099,000", "spec")["refusal_correct"] is None


def test_keyword_recall():
    s = score("Toyota Corolla petrol sedan", "Toyota Corolla is a petrol sedan with 5 seats", "spec")
    assert 0.5 <= s["keyword_recall"] <= 1.0
