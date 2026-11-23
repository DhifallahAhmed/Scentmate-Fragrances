"""clean.py - raw Parfumo CSV -> data/fragrances_clean.json (step 1 of the pipeline).

Usage:  python clean.py
Pipeline:  clean.py -> enrich.py -> train.py -> build_model.py

What it guarantees (the build fails loudly otherwise):
  * every record has a stable `key` (the Parfumo page path, e.g. "Dior/Sauvage_Eau_de_Toilette")
    that survives re-cleaning and re-ordering — feedback is stored against this key, not a row number
  * a missing rating is `null`, never 0 (the source has no real zeros; its smallest count is 2)
  * stray CSV header rows ("Name" / "Main Accords") are dropped
  * every record gets a `productType`: "fragrance" | "aftershave" | "haircare" | "bodycare";
    only "fragrance" is recommended (train.py filters), the rest stay in the data for reference

load(legacy=True) reproduces the OLD cleaner exactly (missing -> 0, header rows kept). It was used once
to map the old row order to keys: data/legacy_index_keys.json.gz, which train.py uses for feedback
collected before keys existed.
"""
import csv, gzip, json, re, sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).parent
SRC = HERE / "data" / "02_Parfumo_Perfumes.csv.gz"
OUT = HERE / "data" / "fragrances_clean.json"

MISSING = {"", "nan", "none", "null", "n/a", "unknown"}
BAD_NAMES = {"unknown", "n/a", "none", "null", "-"}
LIST_COLUMNS = {"Main_Accords": "mainAccords", "Top_Notes": "topNotes", "Middle_Notes": "middleNotes", "Base_Notes": "baseNotes"}

# Product type from concentration + name. Order matters: the first match wins.
PRODUCT_TYPES = [
    ("aftershave", re.compile(r"after[\s-]?shave|apr[eè]s[\s-]?rasage|rasage|shaving", re.I)),
    ("haircare",   re.compile(r"hair (mist|perfume|fragrance|spray)|brume (pour )?(les )?cheveux|parfum (pour )?(les )?cheveux|hair & body", re.I)),
    # Product words only — "soap", "savon", "talc" usually name a SCENT ("Aqua Savon", "Talco"), not a product.
    ("bodycare",   re.compile(r"deodorant|\bdeo\b|shower gel|gel douche|body (lotion|cream|milk|wash|butter|oil)|"
                              r"bath (oil|gel|foam)|lait (corps|parfum)|hand cream", re.I)),
]

def clean_text(value):
    """Normalise whitespace; treat empty / 'nan' / 'unknown' as missing."""
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value).strip())
    return None if value.casefold() in MISSING else value

def clean_list(value):
    """'Rose, Musk, rose' -> ['Rose', 'Musk'] (deduplicated, case-insensitive)."""
    value = clean_text(value)
    out, seen = [], set()
    for item in (value or "").split(","):
        item = clean_text(item)
        if item and item.casefold() not in seen:
            out.append(item); seen.add(item.casefold())
    return out

def clean_year(value):
    try:
        y = int(float(value))
        return y if 1700 <= y <= 2026 else None
    except (TypeError, ValueError, OverflowError):
        return None

def clean_count(value, legacy=False):
    """Number of ratings, or None when the source has no count (NOT 0 — 0 would mean 'nobody rated it')."""
    try:
        n = int(float(value))
        return n if n > 0 else (0 if legacy else None)
    except (TypeError, ValueError, OverflowError):
        return 0 if legacy else None

def page_key(url):
    """'https://www.parfumo.com/Perfumes/Dior/Sauvage_Eau_de_Toilette' -> 'Dior/Sauvage_Eau_de_Toilette'."""
    path = urlparse(url or "").path.strip("/")
    return path.split("/", 1)[1] if path.startswith(("Perfumes/", "Parfums/")) else (path or None)

PERFUME_CONC = re.compile(r"eau de|parfum|perfume|cologne|extrait|attar|esencia|concentrat|toilet water|fragrance|mist|splash|spray", re.I)

def product_type(name, concentration, brand=""):
    """The stated concentration wins ("Eau de Parfum" is a fragrance even from '345 Soap Co.').
    Otherwise look at the name, with the brand removed so brand words don't count."""
    if concentration:
        for label, rx in PRODUCT_TYPES:
            if rx.search(concentration):
                return label
        if PERFUME_CONC.search(concentration):
            return "fragrance"
    text = re.sub(re.escape(brand), " ", name, flags=re.I) if brand else name
    for label, rx in PRODUCT_TYPES:
        if rx.search(text):
            return label
    return "fragrance"

def load(legacy=False):
    rows = list(csv.DictReader(gzip.open(SRC, "rt", encoding="utf-8")))
    records, seen = [], set()
    stats = {"source rows": len(rows), "no name/brand": 0, "header rows": 0, "duplicates": 0}
    for row in rows:
        name, brand = clean_text(row["Name"]), clean_text(row["Brand"])
        if not name or not brand or name.casefold() in BAD_NAMES:
            stats["no name/brand"] += 1
            continue
        if not legacy and name == "Name" and clean_text(row["Main_Accords"]) == "Main Accords":
            stats["header rows"] += 1                      # a stray copy of the CSV header
            continue
        dedup = (name.casefold(), brand.casefold())
        if dedup in seen:
            stats["duplicates"] += 1
            continue
        seen.add(dedup)
        conc = clean_text(row["Concentration"])
        rec = {"name": name, "brand": brand, "releaseYear": clean_year(row["Release_Year"]),
               "concentration": conc.title() if conc else None,
               "ratingCount": clean_count(row["Rating_Count"], legacy)}
        rec.update({dst: clean_list(row[src]) for src, dst in LIST_COLUMNS.items()})
        rec["key"] = page_key(row["URL"])
        if not legacy:
            rec["productType"] = product_type(name, conc, brand)
        records.append(rec)
    if not legacy:
        records = merge_split_pages(records, stats)
    return records, stats

def merge_split_pages(records, stats):
    """A few pages were scraped as two partial rows (one with notes, one with accords/ratings).
    Merge them into one record per page key, keeping the first row's position."""
    by_key, out = {}, []
    for r in records:
        first = by_key.get(r["key"])
        if first is None:
            by_key[r["key"]] = r
            out.append(r)
            continue
        stats["split pages merged"] = stats.get("split pages merged", 0) + 1
        for f in ("mainAccords", "topNotes", "middleNotes", "baseNotes"):
            if not first[f]:
                first[f] = r[f]
        counts = [c for c in (first["ratingCount"], r["ratingCount"]) if c]
        first["ratingCount"] = max(counts) if counts else None
        first["releaseYear"] = first["releaseYear"] or r["releaseYear"]
        first["concentration"] = first["concentration"] or r["concentration"]
    return out

def validate(records):
    """Hard checks — fail the pipeline instead of shipping bad data."""
    keys = [r["key"] for r in records]
    assert all(keys), "every record needs a key"
    dupes = len(keys) - len(set(keys))
    assert dupes == 0, f"{dupes} duplicate keys"
    assert not any(r["ratingCount"] == 0 for r in records), "ratingCount 0 found — missing must be null"
    assert not any(r["name"] == "Name" for r in records), "CSV header row leaked into the data"
    assert not any("Main Accords" in r["mainAccords"] for r in records), "header text in accords"

def main():
    records, stats = load()
    validate(records)
    from collections import Counter
    OUT.write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(stats, "| kept", len(records))
    print("product types:", Counter(r["productType"] for r in records))
    print("ratings: known", sum(r["ratingCount"] is not None for r in records), "| missing (null)", sum(r["ratingCount"] is None for r in records))
    print("written:", OUT)

if __name__ == "__main__":
    main()
