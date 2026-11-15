"""Build the ChromaDB index from cars.csv (one spec card per variant) and markdown docs.

Run: python -m rag.ingest
"""
from pathlib import Path

import pandas as pd

import config
from rag.retriever import COLLECTION, embed, get_collection

CHUNK_WORDS = 375   # ~500 tokens
OVERLAP_WORDS = 40  # ~50 tokens


def spec_card(r) -> str:
    eng = f"{r.engine_cc} cc engine" if str(r.engine_cc) not in ("<NA>", "nan", "") else "electric motor"
    parts = [f"{r.brand} {r.model} {r.variant}: price Rs {int(r.price_pkr):,} PKR (ex-factory, Nov 2026)",
             f"{eng}, {r.fuel}, {r.transmission} transmission, {r.body_type}, {r.seats} seats"]
    if isinstance(r.mileage_kmpl, str) and r.mileage_kmpl:
        parts.append(f"fuel average {r.mileage_kmpl} km/l")
    if isinstance(r.key_features, str) and r.key_features:
        parts.append("features: " + r.key_features.replace(";", ", "))
    return ". ".join(parts) + "."


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if words else []
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def _doc_meta(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    source = next((l[len("Source: "):].strip() for l in lines[:3] if l.startswith("Source: ")), path.name)
    title = lines[0].lstrip("# ").strip() if lines else path.stem
    return title, source


def build_index(cars: pd.DataFrame | None = None, docs_dir: Path | None = None,
                chroma_dir: Path | None = None) -> int:
    """(Re)build the collection. Returns the number of chunks indexed."""
    if cars is None:
        from data.validate import load_cars
        cars = load_cars()
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    chroma_dir = str(chroma_dir or config.CHROMA_DIR)

    ids, texts, metas = [], [], []
    for i, r in enumerate(cars.itertuples()):
        ids.append(f"spec-{i}")
        texts.append(spec_card(r))
        metas.append({"brand": r.brand, "model": r.model, "price_pkr": int(r.price_pkr),
                      "source": r.source_url, "kind": "spec"})
    for path in sorted(docs_dir.glob("*.md")):
        title, source = _doc_meta(path)
        for j, chunk in enumerate(chunk_words(path.read_text(encoding="utf-8"))):
            ids.append(f"doc-{path.stem}-{j}")
            texts.append(chunk)
            metas.append({"brand": "-", "model": "-", "source": source, "kind": "doc", "title": title})

    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    get_collection.cache_clear()
    col = get_collection(chroma_dir)
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embed(texts))
    return len(ids)


if __name__ == "__main__":
    print(f"indexed {build_index()} chunks into {config.CHROMA_DIR}")
