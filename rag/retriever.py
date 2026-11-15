"""Semantic search over the ChromaDB index (heavy imports are lazy)."""
from functools import lru_cache
from pathlib import Path

import config

COLLECTION = "cars"
MIN_SCORE = 0.60  # calibrated: scripts/calibrate_rag.py (relevant >=0.69 typ., off-topic 0.47-0.60)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    if query:
        texts = [QUERY_PREFIX + t for t in texts]
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=4)
def get_collection(chroma_dir: str):
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def retrieve(query: str, k: int = 5, brand: str | None = None, max_price: int | None = None,
             chroma_dir: Path | None = None) -> list[dict]:
    """Return up to k chunks as {text, source, brand, model, score}, best first.

    Chunks scoring below MIN_SCORE (cosine similarity) are dropped, so an off-topic
    query returns []. max_price only keeps spec cards priced at or below it.
    """
    col = get_collection(str(chroma_dir or config.CHROMA_DIR))
    conds = []
    if brand:
        conds.append({"brand": brand})
    if max_price is not None:
        conds.append({"price_pkr": {"$lte": int(max_price)}})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}
    n = min(k, col.count())
    if n == 0:
        return []
    res = col.query(query_embeddings=embed([query], query=True), n_results=n, where=where)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1.0 - dist
        if score >= MIN_SCORE:
            hits.append({"text": text, "source": meta["source"], "brand": meta.get("brand", "-"),
                         "model": meta.get("model", "-"), "score": round(score, 4)})
    return hits
