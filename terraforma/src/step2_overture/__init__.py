"""
Step 2: Download and ingest Overture Maps POI data.

Usage:
    python -m src.step2_overture            # all cities
    python -m src.step2_overture sf         # just San Francisco
"""

import logging
import sys

from src.config import CITY_BBOXES
from src.step2_overture.download import download_city
from src.step2_overture.ingest import ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

ALIASES = {
    "sf": "san_francisco",
    "san_francisco": "san_francisco",
    "nyc": "new_york",
    "new_york": "new_york",
    "paris": "paris",
    "sg": "singapore",
    "singapore": "singapore",
    "london": "london",
    "ldn": "london",
    "mumbai": "mumbai",
    "philly": "philadelphia",
    "philadelphia": "philadelphia",
    "tucson": "tucson",
    "tampa": "tampa",
    "indianapolis": "indianapolis",
    "indy": "indianapolis",
    "nashville": "nashville",
    "new_orleans": "new_orleans",
    "nola": "new_orleans",
    "saint_louis": "saint_louis",
    "stl": "saint_louis",
}


def run(cities: list[str] | None = None):
    targets = cities or list(CITY_BBOXES.keys())
    for city in targets:
        key = ALIASES.get(city, city)
        if key not in CITY_BBOXES:
            print(f"Unknown city: {city}")
            continue
        records = download_city(key)
        ingest(records, key)
        print(f"{key}: {len(records)} Overture places ingested")


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
