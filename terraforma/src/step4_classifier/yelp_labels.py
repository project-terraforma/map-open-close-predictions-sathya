"""
Generate training labels by matching the 3k training dataset to Yelp.

Yelp's `is_closed` field tells us if a business is permanently closed.
We match each training sample by name + coordinates, then save the Yelp
labels so the model can train on confirmed open/closed status.

Usage:
    python -m src.step4_classifier yelp          # label all 3k samples
    python -m src.step4_classifier yelp 1000     # label first 1000 only
"""

import json
import logging
import time

import numpy as np
import pandas as pd
import requests

from src.config import YELP_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent.parent
YELP_LABELS_PATH = PROJECT_ROOT / "yelp_labels.parquet"
TRAINING_DATA_PATH = PROJECT_ROOT / "samples_3k_project_c_updated.parquet"

YELP_SEARCH_URL = "https://api.yelp.com/v3/businesses/search"


def _match_yelp(name: str, location: str, lat: float = None, lon: float = None) -> dict | None:
    """Query Yelp Business Search API to find a business and get its status."""
    if not YELP_API_KEY:
        log.error("YELP_API_KEY not set")
        return None

    try:
        params = {
            "term": name,
            "limit": 1,
        }
        if lat and lon:
            params["latitude"] = lat
            params["longitude"] = lon
            params["radius"] = 200
        else:
            params["location"] = location

        resp = requests.get(YELP_SEARCH_URL, params=params, headers={
            "Authorization": f"Bearer {YELP_API_KEY}",
            "Accept": "application/json",
        }, timeout=10)

        if resp.status_code == 429:
            log.warning("  Rate limited, waiting 5s...")
            time.sleep(5)
            return None
        if resp.status_code != 200:
            log.warning("  Yelp error %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        businesses = data.get("businesses", [])
        if not businesses:
            return None

        biz = businesses[0]
        yelp_name = biz.get("name", "")

        # Basic name overlap check — skip if totally different business
        name_words = set(name.lower().split())
        yelp_words = set(yelp_name.lower().split())
        if name_words and yelp_words:
            overlap = len(name_words & yelp_words) / max(len(name_words), len(yelp_words))
            if overlap < 0.25:
                return None

        is_closed = biz.get("is_closed", False)

        return {
            "yelp_id": biz.get("id"),
            "yelp_name": yelp_name,
            "yelp_is_closed": is_closed,
            "label": 0 if is_closed else 1,
            "status": "CLOSED" if is_closed else "OPEN",
        }

    except Exception as e:
        log.warning("  Yelp error for %s: %s", name, e)
        return None


def generate_labels(limit: int = None):
    """Match 3k training samples to Yelp and save open/closed labels."""
    if not TRAINING_DATA_PATH.exists():
        log.error("Training data not found: %s", TRAINING_DATA_PATH)
        return

    df = pd.read_parquet(TRAINING_DATA_PATH)
    log.info("Loaded %d training samples", len(df))

    if limit:
        df = df.head(limit)
        log.info("Limiting to first %d samples", limit)

    log.info("=" * 60)
    log.info("YELP LABELING: %d training samples", len(df))
    log.info("=" * 60)

    # Load existing labels to skip duplicates
    existing_ids = set()
    if YELP_LABELS_PATH.exists():
        existing = pd.read_parquet(YELP_LABELS_PATH)
        existing_ids = set(existing["id"].values)
        log.info("Existing labels: %d (will skip these)", len(existing_ids))

    new_labels = []
    api_calls = 0
    n_open = 0
    n_closed = 0
    n_agree = 0
    n_disagree = 0

    for idx, row in df.iterrows():
        ov_id = row.get("id")
        name = row.get("names")
        original_label = int(row.get("label", -1))

        # Parse name from JSON if needed
        if isinstance(name, str):
            try:
                parsed = json.loads(name)
                if isinstance(parsed, list) and parsed:
                    first = parsed[0]
                    if isinstance(first, dict):
                        name = first.get("value", name)
                    else:
                        name = str(first)
            except (json.JSONDecodeError, TypeError):
                pass

        # Build location string from addresses (US only — best Yelp coverage)
        addr_raw = row.get("addresses")
        location = ""
        if addr_raw:
            try:
                addrs = json.loads(addr_raw) if isinstance(addr_raw, str) else addr_raw
                if isinstance(addrs, list) and addrs:
                    a = addrs[0]
                    if a.get("country") != "US":
                        continue  # skip non-US
                    parts = []
                    if a.get("freeform"):
                        parts.append(a["freeform"])
                    if a.get("locality"):
                        parts.append(a["locality"])
                    if a.get("region"):
                        parts.append(a["region"])
                    location = ", ".join(parts)
            except (json.JSONDecodeError, TypeError):
                pass

        if not name or not location:
            continue
        if ov_id in existing_ids:
            continue

        api_calls += 1
        result = _match_yelp(str(name), location)

        if not result:
            if api_calls % 100 == 0:
                log.info("  [%d calls, %d labels so far]", api_calls, len(new_labels))
            time.sleep(0.2)
            continue

        # Build record with all Overture fields for feature extraction
        record = {
            "id": ov_id,
            "label": result["label"],
            "original_label": original_label,
            "confidence": row.get("confidence"),
            "yelp_id": result["yelp_id"],
            "yelp_name": result["yelp_name"],
        }

        # Copy Overture fields for feature extraction
        for field in ("sources", "names", "categories", "websites", "socials",
                      "emails", "phones", "brand", "addresses"):
            val = row.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                record[field] = json.dumps(val) if not isinstance(val, str) else val
                record[f"base_{field}"] = record[field]
            else:
                record[field] = None
                record[f"base_{field}"] = None

        record["base_confidence"] = row.get("confidence")
        new_labels.append(record)

        if result["label"] == 1:
            n_open += 1
        else:
            n_closed += 1

        # Track agreement with original labels
        if original_label >= 0:
            if result["label"] == original_label:
                n_agree += 1
            else:
                n_disagree += 1

        if len(new_labels) % 50 == 0:
            log.info("  [%d] %s → Yelp: %s (%s) | Orig: %s | %d open, %d closed | agree: %d, disagree: %d",
                     len(new_labels), str(name)[:30], result["yelp_name"][:20],
                     result["status"],
                     "OPEN" if original_label == 1 else "CLOSED",
                     n_open, n_closed, n_agree, n_disagree)

        time.sleep(0.2)  # rate limit

    if not new_labels:
        log.info("No new labels generated")
        return

    new_df = pd.DataFrame(new_labels)

    # Merge with existing
    if YELP_LABELS_PATH.exists():
        existing = pd.read_parquet(YELP_LABELS_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(YELP_LABELS_PATH, index=False)

    total_open = (combined["label"] == 1).sum()
    total_closed = (combined["label"] == 0).sum()

    log.info("\n" + "=" * 60)
    log.info("YELP LABELING COMPLETE")
    log.info("  New labels this run: %d (%d open, %d closed)", len(new_labels), n_open, n_closed)
    log.info("  Total labels:        %d (%d open, %d closed)", len(combined), total_open, total_closed)
    log.info("  API calls used:      %d", api_calls)
    log.info("  Agreement with original: %d agree, %d disagree (%.1f%% agreement)",
             n_agree, n_disagree,
             100 * n_agree / (n_agree + n_disagree) if (n_agree + n_disagree) > 0 else 0)
    log.info("  Saved to:            %s", YELP_LABELS_PATH.name)
    log.info("=" * 60)
