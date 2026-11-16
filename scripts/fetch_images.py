"""Fetch one freely licensed photo per car model from Wikipedia / Wikimedia Commons.

Writes data/images.csv: brand, model, image_url, file_page, license, artist, wiki_title.
Only images whose Commons licence is free (CC / public domain) are kept, with credits.
Run (needs internet): python -m scripts.fetch_images
"""
import csv
import html
import re
import time

import requests

import config

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "PakCarAssistant-CCP-student-project/1.0 (educational; contact via GitHub ZulqernainAbbasi)"}
FREE = re.compile(r"^(CC|Public domain|PD|CC0|GFDL)", re.I)
OUT = config.DATA_DIR / "images.csv"

# Wikipedia article whose lead photo best represents the model sold in Pakistan.
WIKI_TITLES = {
    ("Suzuki", "Alto"): "Suzuki Alto",
    ("Suzuki", "Every"): "Suzuki Every",
    ("Suzuki", "Cultus"): "Suzuki Celerio",
    ("Suzuki", "Swift"): "Suzuki Swift",
    ("Suzuki", "Fronx"): "Suzuki Fronx",
    ("Toyota", "Yaris"): "Toyota Vios",
    ("Toyota", "Corolla"): "Toyota Corolla (E210)",
    ("Toyota", "Corolla Cross"): "Toyota Corolla Cross",
    ("Toyota", "Hilux"): "Toyota Hilux",
    ("Toyota", "Hilux Revo"): "Toyota Hilux",
    ("Toyota", "Fortuner"): "Toyota Fortuner",
    ("Honda", "City"): "Honda City",
    ("Honda", "Civic"): "Honda Civic (eleventh generation)",
    ("Honda", "BR-V"): "Honda BR-V",
    ("Honda", "HR-V"): "Honda HR-V",
    ("KIA", "Picanto"): "Kia Picanto",
    ("KIA", "Stonic"): "Kia Stonic",
    ("KIA", "Sportage L"): "Kia Sportage",
    ("KIA", "Sorento"): "Kia Sorento",
    ("Hyundai", "Elantra"): "Hyundai Elantra",
    ("Hyundai", "Tucson"): "Hyundai Tucson",
    ("Hyundai", "Santa Fe"): "Hyundai Santa Fe",
    ("Hyundai", "Sonata"): "Hyundai Sonata",
    ("Changan", "Alsvin"): "Changan Alsvin",
    ("Changan", "Karvaan"): "Changan Automobile",
    ("Changan", "Oshan X7"): "Changan Oshan X7",
    ("MG", "ZS"): "MG ZS (SUV)",
    ("MG", "HS"): "MG HS",
    ("MG", "MG4"): "MG4 EV",
    ("MG", "Binguo EV"): "Wuling Bingo",
    ("MG", "ZS EV"): "MG ZS EV",
}


# Models whose article photo is the wrong generation for Pakistan: an exact Commons file
# ("File:...", chosen by hand) or a Commons file search.
FILE_SEARCH = {
    ("Suzuki", "Alto"): "File:Suzuki ALTO S (HA36S) front.JPG",
    ("Toyota", "Corolla"): "Toyota Corolla Altis E210 sedan",
    ("Honda", "City"): "File:2021 Honda City 1.5 GN2 (20211117).jpg",
    ("Changan", "Oshan X7"): "File:Ossan X7 001.jpg",
    ("Suzuki", "Cultus"): "Suzuki Celerio 2015",
    ("Suzuki", "Swift"): "Suzuki Swift 2017",
    ("Toyota", "Yaris"): "Toyota Yaris ATIV",
    ("Honda", "BR-V"): "File:2017 Honda BR-V i-VTEC front.jpg",
    ("Changan", "Karvaan"): "Changan Karvaan",
    ("KIA", "Sportage L"): "Kia Sportage NQ5",
    ("MG", "ZS"): "File:MG ZS (crossover, second generation) DSC 8542.jpg",
}


def _get(params: dict) -> dict:
    """GET the Wikipedia API with polite retries (it returns HTML instead of JSON when rate-limiting)."""
    for attempt in range(4):
        r = requests.get(API, headers=HEADERS, timeout=30, params={"format": "json", **params})
        if r.ok and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        time.sleep(3 * (attempt + 1))
    r.raise_for_status()
    return r.json()


def clean_html(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).replace("\xa0", " ").split())


def parse_page_image(api: dict) -> tuple[str, str] | None:
    for page in api.get("query", {}).get("pages", {}).values():
        if "pageimage" in page and "thumbnail" in page:
            return page["pageimage"], page["thumbnail"]["source"]
    return None


def parse_image_info(api: dict) -> dict | None:
    for page in api.get("query", {}).get("pages", {}).values():
        for info in page.get("imageinfo", []):
            meta = info.get("extmetadata", {})
            lic = clean_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not FREE.match(lic):
                return None
            return {"file_page": info.get("descriptionurl", ""), "license": lic,
                    "artist": clean_html(meta.get("Artist", {}).get("value", "")) or "Unknown"}
    return None


def file_matches_model(file_name: str, model: str) -> bool:
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    tokens = [t for t in model.split() if len(t) > 1] or [model]
    return all(norm(t) in norm(file_name) for t in tokens)


def _file_info(name: str) -> dict | None:
    # en.wikipedia's API also serves Commons file metadata (shared file repository)
    api = _get({"action": "query", "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 800,
                "titles": f"File:{name}"})
    info = parse_image_info(api)
    if not info:
        return None
    thumb = next((ii.get("thumburl") for p in api["query"]["pages"].values() for ii in p.get("imageinfo", [])), None)
    return {"image_url": thumb, **info} if thumb else None


def _search_files(file_query: str, model: str) -> dict | None:
    hits = _get({"action": "query", "list": "search", "srnamespace": 6, "srlimit": 8, "srsearch": file_query})
    for h in hits.get("query", {}).get("search", []):
        name = h["title"].split(":", 1)[1]
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and file_matches_model(name, model):
            info = _file_info(name)
            if info:
                return info
        time.sleep(1)
    return None


def fetch(title: str, file_query: str | None = None, model: str = "") -> dict | None:
    if file_query and file_query.startswith("File:"):
        return _file_info(file_query[5:])
    if file_query:
        got = _search_files(file_query, model)
        if got:
            return got
    found = parse_page_image(_get({"action": "query", "prop": "pageimages", "piprop": "thumbnail|name",
                                   "pithumbsize": 800, "pilicense": "free", "redirects": 1, "titles": title}))
    if found:
        got = _file_info(found[0])
        if got:
            return got
    return _search_files(title if model.lower() in title.lower() else f"{title} {model}", model)


FIELDS = ["brand", "model", "image_url", "file_page", "license", "artist", "wiki_title"]


def main(preview: bool = False, only: list[str] | None = None) -> None:
    existing = {}
    if only and OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            existing = {(r["brand"], r["model"]): r for r in csv.DictReader(f)}
    rows = []
    for (brand, model), title in WIKI_TITLES.items():
        if only and model not in only:
            if (brand, model) in existing:
                rows.append(existing[(brand, model)])
            continue
        try:
            got = fetch(title, FILE_SEARCH.get((brand, model)), model)
        except requests.RequestException as e:
            got = None
            print(f"ERR  {brand} {model}: {type(e).__name__}", flush=True)
        print(f"{'OK  ' if got else 'MISS'} {brand} {model} <- {title} :: "
              f"{got['image_url'].rsplit('/', 1)[-1] if got else '-'}", flush=True)
        if got:
            rows.append({"brand": brand, "model": model, **got, "wiki_title": title})
        elif (brand, model) in existing:
            rows.append(existing[(brand, model)])
        time.sleep(1.5)
    if preview:
        return
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}/{len(WIKI_TITLES)} images -> {OUT}")


if __name__ == "__main__":
    import sys
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    main(preview="--preview" in sys.argv, only=only)
