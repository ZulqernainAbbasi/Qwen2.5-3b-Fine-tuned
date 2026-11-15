import re
import pandas as pd
import pytest

import config
from data.validate import load_cars, model_doc_name, BRANDS

COLUMNS = ["brand", "model", "variant", "price_pkr", "engine_cc", "fuel", "transmission",
           "mileage_kmpl", "body_type", "seats", "key_features", "source_url", "as_of_date"]


@pytest.fixture(scope="module")
def cars():
    return load_cars()


def test_columns_exact(cars):
    assert list(cars.columns) == COLUMNS


def test_all_seven_brands_present(cars):
    assert set(cars.brand) == set(BRANDS)


def test_minimum_coverage(cars):
    assert cars[["brand", "model"]].drop_duplicates().shape[0] >= 25
    assert len(cars) >= 60


def test_price_is_positive_int_between_1m_and_100m(cars):
    assert cars.price_pkr.dtype.kind == "i"
    assert cars.price_pkr.between(1_000_000, 100_000_000).all()


def test_enums_valid(cars):
    assert set(cars.fuel) <= {"petrol", "hybrid", "phev", "electric", "diesel"}
    assert set(cars.body_type) <= {"hatchback", "sedan", "suv", "crossover", "pickup", "mpv"}
    assert set(cars.transmission) <= {"manual", "automatic"}


def test_every_row_has_source_url(cars):
    assert cars.source_url.str.startswith("https://").all()
    assert (cars.as_of_date == "2026-11").all()


def test_no_duplicate_brand_model_variant(cars):
    assert not cars.duplicated(["brand", "model", "variant"]).any()


def test_each_model_has_doc(cars):
    for (brand, model), _ in cars.groupby(["brand", "model"]):
        doc = config.DOCS_DIR / model_doc_name(brand, model)
        assert doc.exists(), doc
        first = doc.read_text(encoding="utf-8").splitlines()
        assert first[0].startswith("# ") and first[1].startswith("Source: ")


def test_load_cars_rejects_bad_rows(tmp_path):
    bad = tmp_path / "cars.csv"
    df = pd.read_csv(config.CARS_CSV)
    df.loc[0, "fuel"] = "steam"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="fuel"):
        load_cars(bad)
