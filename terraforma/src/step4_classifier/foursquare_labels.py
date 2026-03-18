"""
Generate training labels by matching Overture places to Foursquare.

Foursquare's `closed_bucket` field tells us if a business is permanently closed.
We match Overture businesses by name + coordinates, then save the labels
alongside the full Overture raw_json for feature extraction at training time.

Usage:
    python -m src.step4_classifier foursquare sf 500    # label 500 SF businesses
    python -m src.step4_classifier foursquare nyc 500   # label 500 NYC businesses
    python -m src.step4_classifier foursquare all 1000  # label 1000 per city
"""

import json
import logging
import time

import pandas as pd
import requests
from sqlalchemy import text

from src.config import engine, FOURSQUARE_API_KEY, CITY_BBOXES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent.parent
FSQ_LABELS_PATH = PROJECT_ROOT / "foursquare_labels.parquet"

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "chi": "chicago", "chicago": "chicago",
}

FSQ_SEARCH_URL = "https://places-api.foursquare.com/places/search"


def _match_foursquare(name: str, lat: float, lon: float) -> dict | None:
    """Query Foursquare Places API to find a business and get its status."""
    if not FOURSQUARE_API_KEY:
        log.error("FOURSQUARE_API_KEY not set")
        return None

    try:
        resp = requests.get(FSQ_SEARCH_URL, params={
            "query": name,
            "ll": f"{lat},{lon}",
            "radius": 200,
            "limit": 1,
            "fields": "fsq_place_id,name,closed_bucket,location,categories",
        }, headers={
            "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
            "Accept": "application/json",
            "X-Places-Api-Version": "2025-06-17",
        }, timeout=10)

        if resp.status_code == 429:
            log.warning("  Rate limited, waiting 5s...")
            time.sleep(5)
            return None
        if resp.status_code != 200:
            log.warning("  FSQ error %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        place = results[0]
        fsq_name = place.get("name", "")

        # Basic name overlap check — skip if totally different business
        name_words = set(name.lower().split())
        fsq_words = set(fsq_name.lower().split())
        if name_words and fsq_words:
            overlap = len(name_words & fsq_words) / max(len(name_words), len(fsq_words))
            if overlap < 0.25:
                return None

        closed_bucket = place.get("closed_bucket")

        # Map to label: None/absent = open, CLOSED_PERMANENTLY = closed
        if closed_bucket == "CLOSED_PERMANENTLY":
            label = 0
            status = "CLOSED"
        elif closed_bucket == "CLOSED_TEMPORARILY":
            return None  # skip temp closures
        else:
            label = 1
            status = "OPEN"

        return {
            "fsq_id": place.get("fsq_place_id"),
            "fsq_name": fsq_name,
            "label": label,
            "status": status,
        }

    except Exception as e:
        log.warning("  FSQ error for %s: %s", name, e)
        return None


def generate_labels(city: str, limit: int = 500):
    """Match Overture businesses to Foursquare and save open/closed labels."""
    key = ALIASES.get(city, city)
    bbox = CITY_BBOXES.get(key)
    if not bbox:
        log.error("No bbox for city %s", key)
        return

    log.info("=" * 60)
    log.info("FOURSQUARE LABELING: %s (target: %d)", key, limit)
    log.info("=" * 60)

    # Get random Overture businesses
    with engine.connect() as conn:
        candidates = conn.execute(text("""
            SELECT id, name, latitude, longitude, confidence, raw_json
            FROM overture.places
            WHERE latitude BETWEEN :min_lat AND :max_lat
              AND longitude BETWEEN :min_lon AND :max_lon
              AND name IS NOT NULL AND name != ''
            ORDER BY random()
            LIMIT :n
        """), {
            "min_lat": bbox["min_lat"], "max_lat": bbox["max_lat"],
            "min_lon": bbox["min_lon"], "max_lon": bbox["max_lon"],
            "n": limit * 3,  # oversample since some won't match
        }).fetchall()

    log.info("Candidate pool: %d Overture places", len(candidates))

    # Load existing labels to skip duplicates
    existing_ids = set()
    if FSQ_LABELS_PATH.exists():
        existing = pd.read_parquet(FSQ_LABELS_PATH)
        existing_ids = set(existing["id"].values)
        log.info("Existing labels: %d (will skip these)", len(existing_ids))

    new_labels = []
    api_calls = 0
    n_open = 0
    n_closed = 0

    for ov_id, name, lat, lon, confidence, raw_json in candidates:
        if len(new_labels) >= limit:
            break
        if ov_id in existing_ids:
            continue

        api_calls += 1
        result = _match_foursquare(name, lat, lon)

        if not result:
            if api_calls % 50 == 0:
                log.info("  [%d calls, %d labels so far]", api_calls, len(new_labels))
            time.sleep(0.1)
            continue

        # Parse Overture raw_json
        rj = raw_json
        if isinstance(rj, str):
            try:
                rj = json.loads(rj)
            except (ValueError, TypeError):
                rj = {}

        record = {
            "id": ov_id,
            "label": result["label"],
            "city": key,
            "confidence": confidence,
            "fsq_id": result["fsq_id"],
            "fsq_name": result["fsq_name"],
        }

        # Store Overture fields as JSON strings for feature extraction
        if isinstance(rj, dict):
            for field in ("sources", "names", "categories", "websites", "socials",
                          "emails", "phones", "brand", "addresses"):
                val = rj.get(field)
                if val is not None:
                    record[field] = json.dumps(val) if not isinstance(val, str) else val
                    record[f"base_{field}"] = record[field]
                else:
                    record[field] = None
                    record[f"base_{field}"] = None

        record["base_confidence"] = confidence
        new_labels.append(record)

        if result["label"] == 1:
            n_open += 1
        else:
            n_closed += 1

        if len(new_labels) % 25 == 0:
            log.info("  [%d] %s → FSQ: %s (%s) | Total: %d open, %d closed",
                     len(new_labels), name[:35], result["fsq_name"][:25],
                     result["status"], n_open, n_closed)

        time.sleep(0.1)  # rate limit

    if not new_labels:
        log.info("No new labels generated")
        return

    new_df = pd.DataFrame(new_labels)

    # Merge with existing
    if FSQ_LABELS_PATH.exists():
        existing = pd.read_parquet(FSQ_LABELS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(FSQ_LABELS_PATH, index=False)

    total_open = (combined["label"] == 1).sum()
    total_closed = (combined["label"] == 0).sum()

    log.info("\n" + "=" * 60)
    log.info("FOURSQUARE LABELING COMPLETE: %s", key)
    log.info("  New labels this run: %d (%d open, %d closed)", len(new_labels), n_open, n_closed)
    log.info("  Total labels:        %d (%d open, %d closed)", len(combined), total_open, total_closed)
    log.info("  API calls used:      %d", api_calls)
    log.info("  Saved to:            %s", FSQ_LABELS_PATH.name)
    log.info("=" * 60)

    # Per-city breakdown
    if "city" in combined.columns:
        log.info("\nPer-city breakdown:")
        for c in sorted(combined["city"].unique()):
            subset = combined[combined["city"] == c]
            log.info("  %s: %d (%d open, %d closed)",
                     c, len(subset),
                     (subset["label"] == 1).sum(),
                     (subset["label"] == 0).sum())
