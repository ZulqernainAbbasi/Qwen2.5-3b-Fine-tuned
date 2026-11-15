"""Print top retrieval scores for relevant vs off-topic queries to choose MIN_SCORE.

Run: python -m scripts.calibrate_rag
"""
import rag.retriever as r
from rag.ingest import build_index

print("indexed", build_index())
r.MIN_SCORE = -1.0
QUERIES = {
    "relevant": ["Toyota Corolla price", "cheapest automatic car", "hybrid SUV fuel average",
                 "how much down payment for car loan", "Alto ki qeemat kya hai", "7 seater family car",
                 "electric car range", "freight charges for Toyota in Lahore", "oil change interval",
                 "Honda Civic RS features"],
    "offtopic": ["recipe for chicken biryani with saffron", "who won the cricket world cup",
                 "Tesla Model 3 price", "how to learn python programming", "weather in Karachi tomorrow",
                 "best laptop under 1 lakh", "Mercedes C200 specs", "iPhone 17 price in Pakistan"],
}
for group, qs in QUERIES.items():
    for q in qs:
        hits = r.retrieve(q, k=1)
        print(f"{group:9s} {hits[0]['score']:.3f}  {q}  ->  {hits[0]['text'][:60]!r}")
