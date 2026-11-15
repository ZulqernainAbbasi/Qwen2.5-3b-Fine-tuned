"""Project-wide paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CARS_CSV = DATA_DIR / "cars.csv"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "bookings.db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = os.getenv("LLM_MODEL", "car-assistant")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

REFUSAL = "I don't have verified information on that."
PRICE_DISCLAIMER = "Prices are ex-factory as of Nov 2026, verify with the dealer."
