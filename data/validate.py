"""Load and validate the car dataset."""
from pathlib import Path

import pandas as pd

import config

BRANDS = ["Toyota", "Honda", "Suzuki", "KIA", "Hyundai", "Changan", "MG"]
FUELS = {"petrol", "hybrid", "phev", "electric", "diesel"}
BODIES = {"hatchback", "sedan", "suv", "crossover", "pickup", "mpv"}
TRANSMISSIONS = {"manual", "automatic"}


def model_doc_name(brand: str, model: str) -> str:
    slug = f"{brand}-{model}".lower().replace(" ", "-").replace(":", "").replace("+", "plus")
    return f"{slug}.md"


def load_cars(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or config.CARS_CSV, dtype={"as_of_date": str, "mileage_kmpl": str})
    df["key_features"] = df["key_features"].fillna("")
    df["mileage_kmpl"] = df["mileage_kmpl"].fillna("")
    errors = []
    checks = {
        "brand": ~df.brand.isin(BRANDS),
        "fuel": ~df.fuel.isin(FUELS),
        "body_type": ~df.body_type.isin(BODIES),
        "transmission": ~df.transmission.isin(TRANSMISSIONS),
        "price_pkr": ~pd.to_numeric(df.price_pkr, errors="coerce").between(1_000_000, 100_000_000),
        "source_url": ~df.source_url.astype(str).str.startswith("https://"),
    }
    for col, bad in checks.items():
        for i in df.index[bad]:
            errors.append(f"row {i + 2}: bad {col} = {df.at[i, col]!r}")
    if errors:
        raise ValueError("Invalid cars.csv:\n" + "\n".join(errors))
    df["price_pkr"] = df["price_pkr"].astype("int64")
    df["engine_cc"] = df["engine_cc"].astype("Int64")
    df["seats"] = df["seats"].astype("int64")
    return df
