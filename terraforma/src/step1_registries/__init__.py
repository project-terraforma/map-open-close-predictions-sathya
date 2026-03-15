"""
Step 1: Download and ingest city business registry data.

Usage:
    python -m src.step1_registries            # all cities
    python -m src.step1_registries sf          # just San Francisco
    python -m src.step1_registries nyc         # just New York
"""

import logging
import sys

from src.step1_registries.download import DOWNLOADERS
from src.step1_registries.ingest import ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

ALIASES = {
    "sf": "san_francisco",
    "san_francisco": "san_francisco",
    "nyc": "new_york",
    "new_york": "new_york",
    "chi": "chicago",
    "chicago": "chicago",
    "paris": "paris",
    "sg": "singapore",
    "singapore": "singapore",
    "london": "london",
    "ldn": "london",
    "mumbai": "mumbai",
}


def run(cities: list[str] | None = None):
    targets = cities or list(DOWNLOADERS.keys())
    for city in targets:
        key = ALIASES.get(city, city)
        downloader = DOWNLOADERS.get(key)
        if not downloader:
            print(f"Unknown city: {city}")
            continue
        records = downloader()
        ingest(records, key)
        print(f"{key}: {len(records)} records ingested")


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
