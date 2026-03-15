"""
Match Yelp Open Dataset businesses to Overture places and create training data.

Usage:
    python -m src.step4_classifier.yelp_training match    # match Yelp → Overture
    python -m src.step4_classifier.yelp_training stats    # show match stats
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YELP_FILE = PROJECT_ROOT / "Yelp JSON" / "yelp_academic_dataset_business.json"
OUTPUT_FILE = PROJECT_ROOT / "yelp_overture_training.parquet"


YELP_TO_OVERTURE_CITY = {
    "Philadelphia": "philadelphia",
    "Tucson": "tucson",
    "Tampa": "tampa",
    "Indianapolis": "indianapolis",
    "Nashville": "nashville",
    "New Orleans": "new_orleans",
    "Saint Louis": "saint_louis",
    "St. Louis": "saint_louis",
}


def load_yelp(cities=None):
    """Load Yelp businesses for specified cities."""
    if cities is None:
        cities = set(YELP_TO_OVERTURE_CITY.keys())

    businesses = []
    with open(YELP_FILE, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b.get("city") in cities:
                businesses.append({
                    "yelp_id": b["business_id"],
                    "yelp_name": b["name"],
                    "yelp_city": b["city"],
                    "yelp_lat": b["latitude"],
                    "yelp_lon": b["longitude"],
                    "yelp_address": b.get("address", ""),
                    "yelp_categories": b.get("categories", ""),
                    "yelp_stars": b.get("stars", 0),
                    "yelp_review_count": b.get("review_count", 0),
                    "is_open": b["is_open"],
                })

    log.info("Loaded %d Yelp businesses from %s", len(businesses), ", ".join(cities))
    return pd.DataFrame(businesses)


def match_yelp_to_overture(yelp_df):
    """Match Yelp businesses to Overture places by name + proximity."""
    from src.config import engine
    from sqlalchemy import text
    from scipy.spatial import cKDTree

    # Group Yelp by Overture city name
    yelp_df = yelp_df.copy()
    yelp_df["overture_city"] = yelp_df["yelp_city"].map(YELP_TO_OVERTURE_CITY)
    overture_cities = yelp_df["overture_city"].dropna().unique().tolist()

    if not overture_cities:
        log.error("No Yelp cities mapped to Overture cities")
        return pd.DataFrame()

    all_matches = []

    for ov_city in overture_cities:
        city_yelp = yelp_df[yelp_df["overture_city"] == ov_city]
        log.info("\n--- %s: %d Yelp businesses ---", ov_city, len(city_yelp))

        with engine.connect() as conn:
            overture_rows = conn.execute(text("""
                SELECT id, name, latitude, longitude, confidence,
                       website, phone, address, category, city, raw_json
                FROM overture.places
                WHERE city = :city
            """), {"city": ov_city}).fetchall()

        if not overture_rows:
            log.warning("No Overture data for %s. Run: python -m src.step2_overture %s",
                        ov_city, ov_city)
            continue

        log.info("  %d Overture places loaded", len(overture_rows))

        # Build overture lookup
        overture = []
        for r in overture_rows:
            raw = r[10] if isinstance(r[10], dict) else (json.loads(r[10]) if r[10] else {})
            overture.append({
                "overture_id": r[0], "o_name": r[1], "o_lat": r[2], "o_lon": r[3],
                "confidence": r[4], "sources": json.dumps(raw.get("sources", [])),
                "website": r[5], "phone": r[6], "address": r[7], "category": r[8],
                "city": r[9],
            })
        ov_df = pd.DataFrame(overture)

        # Build spatial index
        ov_coords = np.array(list(zip(ov_df["o_lat"], ov_df["o_lon"])))
        tree = cKDTree(ov_coords)

        matches = []
        matched_overture_ids = set()
        checked = 0

        for _, yelp in city_yelp.iterrows():
            checked += 1
            if checked % 2000 == 0:
                log.info("  checked %d / %d (matched %d so far)",
                         checked, len(city_yelp), len(matches))

            nearby_idx = tree.query_ball_point([yelp["yelp_lat"], yelp["yelp_lon"]], r=0.005)
            if not nearby_idx:
                continue

            yelp_name_lower = yelp["yelp_name"].lower().strip()
            best_sim = 0
            best_idx = None

            for idx in nearby_idx:
                o_name = ov_df.iloc[idx]["o_name"] or ""
                sim = SequenceMatcher(None, yelp_name_lower, o_name.lower().strip()).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

            if best_sim >= 0.70 and best_idx is not None:
                best = ov_df.iloc[best_idx]
                if best["overture_id"] not in matched_overture_ids:
                    matched_overture_ids.add(best["overture_id"])
                    matches.append({
                        "overture_id": best["overture_id"],
                        "name": best["o_name"],
                        "latitude": best["o_lat"],
                        "longitude": best["o_lon"],
                        "confidence": best["confidence"],
                        "sources": best["sources"],
                        "website": best["website"],
                        "phone": best["phone"],
                        "address": best["address"],
                        "category": best["category"],
                        "city": best["city"],
                        "yelp_name": yelp["yelp_name"],
                        "yelp_categories": yelp["yelp_categories"],
                        "yelp_stars": yelp["yelp_stars"],
                        "yelp_review_count": yelp["yelp_review_count"],
                        "name_sim": best_sim,
                        "is_open": yelp["is_open"],
                    })

        log.info("  %s: matched %d / %d (%.1f%%)",
                 ov_city, len(matches), len(city_yelp),
                 100 * len(matches) / len(city_yelp) if len(city_yelp) > 0 else 0)
        all_matches.extend(matches)

    match_df = pd.DataFrame(all_matches)
    log.info("\n=== TOTAL: matched %d / %d Yelp businesses (%.1f%%) ===",
             len(match_df), len(yelp_df), 100 * len(match_df) / len(yelp_df))

    if len(match_df) > 0:
        open_pct = 100 * match_df["is_open"].mean()
        log.info("  %d open (%.0f%%), %d closed (%.0f%%)",
                 match_df["is_open"].sum(), open_pct,
                 len(match_df) - match_df["is_open"].sum(), 100 - open_pct)

    return match_df


def save_training_data(match_df):
    """Save matched data as parquet for training."""
    match_df.to_parquet(OUTPUT_FILE, index=False)
    log.info("Saved training data to %s (%d rows)", OUTPUT_FILE.name, len(match_df))


def show_stats():
    """Show stats on existing matched data."""
    if not OUTPUT_FILE.exists():
        log.error("No training data found. Run: python -m src.step4_classifier.yelp_training match")
        return

    df = pd.read_parquet(OUTPUT_FILE)
    log.info("Yelp-Overture training data: %d rows", len(df))
    log.info("  Open: %d (%.0f%%)", df["is_open"].sum(), 100 * df["is_open"].mean())
    log.info("  Closed: %d (%.0f%%)", len(df) - df["is_open"].sum(),
             100 * (1 - df["is_open"].mean()))
    log.info("  Avg name similarity: %.2f", df["name_sim"].mean())
    log.info("  Avg confidence: %.2f", df["confidence"].mean())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "match"

    if cmd == "match":
        yelp_df = load_yelp()
        match_df = match_yelp_to_overture(yelp_df)
        if len(match_df) > 0:
            save_training_data(match_df)
    elif cmd == "stats":
        show_stats()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m src.step4_classifier.yelp_training [match|stats]")
