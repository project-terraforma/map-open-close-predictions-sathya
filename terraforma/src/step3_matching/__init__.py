"""
Step 3: Match registry businesses to Overture places.

Usage:
    python -m src.step3_matching            # all cities
    python -m src.step3_matching sf         # just San Francisco
"""

import logging
import sys

from src.step3_matching.match import match_city
from src.step3_matching.ingest import ingest

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

# US-only: focus training and prediction on these cities
MATCHABLE_CITIES = ["san_francisco", "new_york", "chicago"]


def run(cities: list[str] | None = None):
    targets = cities or MATCHABLE_CITIES
    for city in targets:
        key = ALIASES.get(city, city)
        matches = match_city(key)
        ingest(matches, key)
        print(f"{key}: {len(matches)} matches ingested")


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
