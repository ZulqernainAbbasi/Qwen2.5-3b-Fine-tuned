import pandas as pd
import pytest

from rag.ingest import build_index
from rag.retriever import retrieve

ROWS = [
    dict(brand="Toyota", model="Corolla", variant="Altis 1.6 MT", price_pkr=6099000, engine_cc=1598,
         fuel="petrol", transmission="manual", mileage_kmpl="", body_type="sedan", seats=5,
         key_features="", source_url="https://example.com/corolla", as_of_date="2026-11"),
    dict(brand="Suzuki", model="Alto", variant="VXR", price_pkr=2994861, engine_cc=658,
         fuel="petrol", transmission="manual", mileage_kmpl="16-22", body_type="hatchback", seats=4,
         key_features="2 airbags;ABS", source_url="https://example.com/alto", as_of_date="2026-11"),
    dict(brand="Hyundai", model="Tucson", variant="Hybrid Smart FWD", price_pkr=12202000, engine_cc=1598,
         fuel="hybrid", transmission="automatic", mileage_kmpl="14-17", body_type="suv", seats=5,
         key_features="", source_url="https://example.com/tucson", as_of_date="2026-11"),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    root = tmp_path_factory.mktemp("rag")
    docs = root / "docs"
    docs.mkdir()
    (docs / "financing-pakistan.md").write_text(
        "# Car Financing in Pakistan\nSource: https://example.com/fin\n\n"
        "Banks offer auto loans for 1 to 7 years with a minimum down payment of 30 percent.\n",
        encoding="utf-8")
    chroma = root / "chroma"
    n = build_index(cars=pd.DataFrame(ROWS), docs_dir=docs, chroma_dir=chroma)
    return chroma, n


def test_build_index_counts(index):
    _, n = index
    assert n == 4  # 3 spec cards + 1 doc chunk


def test_retrieve_finds_corolla_for_corolla_price(index):
    chroma, _ = index
    hits = retrieve("What is the price of Toyota Corolla?", chroma_dir=chroma)
    assert hits and hits[0]["model"] == "Corolla"
    assert "6,099,000" in hits[0]["text"]
    assert set(hits[0]) == {"text", "source", "brand", "model", "score"}


def test_brand_filter(index):
    chroma, _ = index
    hits = retrieve("cheap car", brand="Suzuki", chroma_dir=chroma)
    assert hits and all(h["brand"] == "Suzuki" for h in hits)


def test_max_price_filter_excludes_expensive(index):
    chroma, _ = index
    hits = retrieve("family SUV", max_price=7_000_000, chroma_dir=chroma)
    assert all(h["model"] != "Tucson" for h in hits)


def test_doc_chunk_retrievable(index):
    chroma, _ = index
    hits = retrieve("how much down payment for a car loan", chroma_dir=chroma)
    assert hits and hits[0]["source"] == "https://example.com/fin"


def test_irrelevant_query_returns_empty(index):
    chroma, _ = index
    assert retrieve("recipe for chicken biryani with saffron", chroma_dir=chroma) == []
