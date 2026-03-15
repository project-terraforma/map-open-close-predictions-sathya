"""
Build training data via FULL OUTER JOIN of two Overture releases across 12 US cities.

Adopted from StatusNow approach:
  - Closed: existed in Jan release but GONE in Feb (churned)
  - Open: exists in Feb release (either new or still present)
  - Both snapshots preserved for delta feature computation

Usage:
    python -m src.step4_classifier.delta_features build          # 12 cities, 3k/class/city
    python -m src.step4_classifier.delta_features build 5000     # custom limit per class per city
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RELEASE_JAN = "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*"
RELEASE_FEB = "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/*"

OUTPUT_FILE = PROJECT_ROOT / "delta_training.parquet"

# 12 US cities with bounding boxes (from config.py)
TRAINING_CITIES = {
    "san_francisco": {"min_lon": -122.52, "max_lon": -122.35, "min_lat": 37.70, "max_lat": 37.82},
    "new_york":      {"min_lon": -74.05,  "max_lon": -73.90,  "min_lat": 40.68, "max_lat": 40.88},
    "chicago":       {"min_lon": -87.94,  "max_lon": -87.52,  "min_lat": 41.64, "max_lat": 42.02},
    "philadelphia":  {"min_lon": -75.28,  "max_lon": -75.10,  "min_lat": 39.87, "max_lat": 40.02},
    "tucson":        {"min_lon": -111.10, "max_lon": -110.75, "min_lat": 32.05, "max_lat": 32.35},
    "tampa":         {"min_lon": -82.55,  "max_lon": -82.35,  "min_lat": 27.85, "max_lat": 28.10},
    "indianapolis":  {"min_lon": -86.30,  "max_lon": -86.05,  "min_lat": 39.65, "max_lat": 39.90},
    "nashville":     {"min_lon": -86.90,  "max_lon": -86.65,  "min_lat": 36.05, "max_lat": 36.25},
    "new_orleans":   {"min_lon": -90.15,  "max_lon": -89.95,  "min_lat": 29.90, "max_lat": 30.05},
    "houston":       {"min_lon": -95.60,  "max_lon": -95.20,  "min_lat": 29.60, "max_lat": 29.90},
    "phoenix":       {"min_lon": -112.20, "max_lon": -111.90, "min_lat": 33.30, "max_lat": 33.60},
    "atlanta":       {"min_lon": -84.50,  "max_lon": -84.30,  "min_lat": 33.70, "max_lat": 33.85},
}

METADATA_COLS = """
    p.id,
    p.confidence,
    to_json(p.sources) AS sources,
    to_json(p.names) AS names,
    to_json(p.categories) AS categories,
    to_json(p.websites) AS websites,
    to_json(p.phones) AS phones,
    to_json(p.emails) AS emails,
    to_json(p.socials) AS socials,
    to_json(p.brand) AS brand,
    to_json(p.addresses) AS addresses
"""


def _fetch_city(con, release_url, city, bbox):
    """Fetch all places for a city bbox from one Overture release."""
    query = f"""
        SELECT {METADATA_COLS}
        FROM read_parquet('{release_url}', hive_partitioning=true) p
        WHERE p.bbox.xmin >= {bbox['min_lon']}
          AND p.bbox.xmax <= {bbox['max_lon']}
          AND p.bbox.ymin >= {bbox['min_lat']}
          AND p.bbox.ymax <= {bbox['max_lat']}
    """
    return con.execute(query).fetchdf()


def build(max_per_class_per_city: int = 3000):
    """Build truth dataset via FULL OUTER JOIN across 12 cities."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'")

    all_city_dfs = []

    for city, bbox in TRAINING_CITIES.items():
        log.info("\n=== %s ===", city.upper())

        # Cache per-city parquets to avoid re-downloading
        cache_jan = PROJECT_ROOT / f"cache_{city}_jan.parquet"
        cache_feb = PROJECT_ROOT / f"cache_{city}_feb.parquet"

        if cache_jan.exists():
            jan_df = pd.read_parquet(cache_jan)
            log.info("  Jan: %d (cached)", len(jan_df))
        else:
            log.info("  Fetching Jan release... (slow, ~2-5 min per city)")
            jan_df = _fetch_city(con, RELEASE_JAN, city, bbox)
            jan_df.to_parquet(cache_jan, index=False)
            log.info("  Jan: %d places", len(jan_df))

        if cache_feb.exists():
            feb_df = pd.read_parquet(cache_feb)
            log.info("  Feb: %d (cached)", len(feb_df))
        else:
            log.info("  Fetching Feb release... (slow, ~2-5 min per city)")
            feb_df = _fetch_city(con, RELEASE_FEB, city, bbox)
            feb_df.to_parquet(cache_feb, index=False)
            log.info("  Feb: %d places", len(feb_df))

        # FULL OUTER JOIN on id
        jan_ids = set(jan_df["id"].values)
        feb_ids = set(feb_df["id"].values)
        jan_indexed = jan_df.set_index("id")
        feb_indexed = feb_df.set_index("id")

        churned_ids = jan_ids - feb_ids  # closed: gone in Feb
        new_ids = feb_ids - jan_ids       # new: appeared in Feb
        both_ids = jan_ids & feb_ids      # still open

        log.info("  Churned (closed): %d | New: %d | Both: %d",
                 len(churned_ids), len(new_ids), len(both_ids))

        records = []

        # Closed businesses (churned): use Jan as "base" and current snapshot
        for oid in churned_ids:
            j = jan_indexed.loc[oid]
            if isinstance(j, pd.DataFrame):
                j = j.iloc[0]
            r = {"id": oid, "label": 0, "city": city}
            # Base = Jan (last snapshot before closure)
            # Current = COALESCE(Feb, Jan) = Jan (since they're gone from Feb)
            # This matches StatusNow: churned places have current == base
            for col in jan_indexed.columns:
                r[f"base_{col}"] = j.get(col)
                r[col] = j.get(col)  # current = base for churned
            records.append(r)

        # Open businesses (still in both): use Jan as base, Feb as current
        for oid in both_ids:
            j = jan_indexed.loc[oid]
            f = feb_indexed.loc[oid]
            if isinstance(j, pd.DataFrame):
                j = j.iloc[0]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            r = {"id": oid, "label": 1, "city": city}
            for col in jan_indexed.columns:
                r[f"base_{col}"] = j.get(col)
                r[col] = f.get(col)  # current = Feb
            records.append(r)

        # New businesses (appeared in Feb): use Feb as both base and current
        for oid in new_ids:
            f = feb_indexed.loc[oid]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            r = {"id": oid, "label": 1, "city": city}
            for col in feb_indexed.columns:
                r[f"base_{col}"] = f.get(col)  # no Jan data, use Feb as base
                r[col] = f.get(col)
            records.append(r)

        city_df = pd.DataFrame(records)

        # Downsample per city to balance and limit size
        closed = city_df[city_df["label"] == 0]
        opened = city_df[city_df["label"] == 1]

        n_closed = min(len(closed), max_per_class_per_city)
        n_open = min(len(opened), max_per_class_per_city)

        if n_closed > 0:
            closed = closed.sample(n=n_closed, random_state=42)
        if n_open > 0:
            opened = opened.sample(n=n_open, random_state=42)

        city_df = pd.concat([closed, opened], ignore_index=True)
        log.info("  Sampled: %d closed, %d open", n_closed, len(city_df) - n_closed)
        all_city_dfs.append(city_df)

    con.close()

    # Combine all cities
    result = pd.concat(all_city_dfs, ignore_index=True)

    n_open = (result["label"] == 1).sum()
    n_closed = (result["label"] == 0).sum()

    result.to_parquet(OUTPUT_FILE, index=False)
    log.info("\n" + "=" * 60)
    log.info("Saved %d rows to %s", len(result), OUTPUT_FILE.name)
    log.info("  Open: %d (%.0f%%)", n_open, 100 * n_open / len(result))
    log.info("  Closed: %d (%.0f%%)", n_closed, 100 * n_closed / len(result))

    for c in sorted(result["city"].unique()):
        mask = result["city"] == c
        c_open = (result.loc[mask, "label"] == 1).sum()
        c_closed = (result.loc[mask, "label"] == 0).sum()
        log.info("  %s: %d open, %d closed", c, c_open, c_closed)

    return result


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        build(max_per_class_per_city=lim)
    else:
        print("Usage: python -m src.step4_classifier.delta_features build [limit_per_class_per_city]")
