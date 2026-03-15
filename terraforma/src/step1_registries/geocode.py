"""
Geocode registry businesses that are missing lat/lon using Nominatim (OpenStreetMap).

Rate limited to 1 request/sec per Nominatim usage policy.
Saves progress after every successful geocode so you can Ctrl+C and resume.
"""

import logging
import time

import requests
from sqlalchemy import text

from src.config import engine, CITY_BBOXES

log = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "terraforma-research/1.0"}

CITY_CONTEXT = {
    "san_francisco": "San Francisco, CA, USA",
    "new_york": "New York, NY, USA",
    "paris": "Paris, France",
    "singapore": "Singapore",
    "london": "London, United Kingdom",
    "mumbai": "Mumbai, Maharashtra, India",
}


def geocode_city(city: str):
    """Geocode all businesses in a city that lack coordinates."""
    bbox = CITY_BBOXES.get(city)
    if not bbox:
        log.error("Unknown city: %s", city)
        return

    context = CITY_CONTEXT.get(city, city)

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, address, name
            FROM registries.businesses
            WHERE city = :city AND latitude IS NULL AND address IS NOT NULL
            ORDER BY id
        """), {"city": city}).fetchall()

    total = len(rows)
    log.info("Geocoding %d businesses for %s", total, city)

    if total == 0:
        log.info("Nothing to geocode for %s", city)
        return

    success = 0
    failed = 0

    for i, (biz_id, address, name) in enumerate(rows):
        query = f"{address}, {context}"

        try:
            resp = requests.get(NOMINATIM_URL, params={
                "q": query,
                "format": "json",
                "limit": 1,
                "viewbox": f"{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']},{bbox['min_lat']}",
                "bounded": 1,
            }, headers=HEADERS, timeout=10)

            results = resp.json()

            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE registries.businesses
                        SET latitude = :lat, longitude = :lon
                        WHERE id = :id
                    """), {"lat": lat, "lon": lon, "id": biz_id})
                success += 1
            else:
                failed += 1

        except Exception as e:
            log.warning("Geocode error for id=%d: %s", biz_id, e)
            failed += 1

        if (i + 1) % 100 == 0:
            log.info("  %d / %d geocoded (%d success, %d failed)", i + 1, total, success, failed)

        time.sleep(1.1)  # respect Nominatim rate limit

    log.info("Done geocoding %s: %d success, %d failed out of %d",
             city, success, failed, total)
