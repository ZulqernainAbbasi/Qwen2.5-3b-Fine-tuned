from frontend.catalog import fmt_pkr, load_models, models_in_text

import config


def _models():
    return load_models(config.ROOT)


def test_load_models_groups_variants():
    ms = {(m["brand"], m["model"]): m for m in _models()}
    yaris = ms[("Toyota", "Yaris")]
    assert yaris["min_price"] == 4_649_000 and yaris["max_price"] == 6_389_000
    assert len(yaris["variants"]) == 5
    assert "sedan" in yaris["body_types"]


def test_models_in_text_prefers_longest_name():
    found = [(m["brand"], m["model"]) for m in models_in_text("Try the Toyota Corolla Cross 1.8 HEV", _models())]
    assert found == [("Toyota", "Corolla Cross")]


def test_models_in_text_multiple_in_order():
    text = "- Suzuki Alto VXR: Rs 2,994,861\n- Honda City 1.2L CVT: Rs 4,737,000"
    found = [(m["brand"], m["model"]) for m in models_in_text(text, _models())]
    assert found == [("Suzuki", "Alto"), ("Honda", "City")]


def test_common_words_need_brand():
    assert models_in_text("great for city driving every day", _models()) == []


def test_short_names_need_brand():
    found = [(m["brand"], m["model"]) for m in models_in_text("The MG ZS EV and MG HS are good", _models())]
    assert found == [("MG", "ZS EV"), ("MG", "HS")]


def test_fmt_pkr():
    assert fmt_pkr(4_649_000) == "46.5 lakh"
    assert fmt_pkr(10_499_000) == "1.05 crore"
    assert fmt_pkr(2_994_861) == "29.9 lakh"
