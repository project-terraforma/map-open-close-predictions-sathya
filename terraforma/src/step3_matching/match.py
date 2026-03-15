"""
Match registry businesses to Overture places using name similarity + geographic proximity.
"""

import logging
import math

from rapidfuzz import fuzz
from sqlalchemy import text

from src.config import engine

log = logging.getLogger(__name__)

# Matching thresholds
NAME_THRESHOLD = 70       # minimum fuzzy name match score (0-100)
MAX_DISTANCE_M = 200      # maximum distance in meters between registry and Overture
COMBINED_THRESHOLD = 60   # minimum combined score to accept a match


def _haversine_m(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _name_score(name1: str | None, name2: str | None) -> float:
    if not name1 or not name2:
        return 0.0
    return fuzz.token_sort_ratio(name1, name2)


def _geo_score(dist_m: float) -> float:
    """Convert distance to a 0-100 score (100 = same location, 0 = MAX_DISTANCE_M away)."""
    if dist_m >= MAX_DISTANCE_M:
        return 0.0
    return 100.0 * (1.0 - dist_m / MAX_DISTANCE_M)


def match_city(city: str) -> list[dict]:
    """For each geocoded registry business, find the best matching Overture place."""
    log.info("Matching registry → Overture for %s", city)

    with engine.connect() as conn:
        # Load registry businesses with coordinates
        reg_rows = conn.execute(text("""
            SELECT id, name_normalized, latitude, longitude, is_open
            FROM registries.businesses
            WHERE city = :city AND latitude IS NOT NULL AND longitude IS NOT NULL
        """), {"city": city}).fetchall()

        log.info("  %d geocoded registry businesses", len(reg_rows))

        # For each registry business, find nearby Overture places using PostGIS
        matches = []
        batch_size = 500
        for i in range(0, len(reg_rows), batch_size):
            batch = reg_rows[i:i + batch_size]
            for reg_id, reg_name, reg_lat, reg_lon, reg_is_open in batch:
                # Find Overture places within MAX_DISTANCE_M
                candidates = conn.execute(text("""
                    SELECT id, name_normalized, latitude, longitude
                    FROM overture.places
                    WHERE city = :city
                      AND latitude BETWEEN :min_lat AND :max_lat
                      AND longitude BETWEEN :min_lon AND :max_lon
                """), {
                    "city": city,
                    "min_lat": reg_lat - 0.002,  # ~200m latitude
                    "max_lat": reg_lat + 0.002,
                    "min_lon": reg_lon - 0.003,  # ~200m longitude (varies by latitude)
                    "max_lon": reg_lon + 0.003,
                }).fetchall()

                best_score = 0
                best_match = None
                for ov_id, ov_name, ov_lat, ov_lon in candidates:
                    ns = _name_score(reg_name, ov_name)
                    if ns < NAME_THRESHOLD:
                        continue
                    dist = _haversine_m(reg_lat, reg_lon, ov_lat, ov_lon)
                    if dist > MAX_DISTANCE_M:
                        continue
                    gs = _geo_score(dist)
                    combined = 0.7 * ns + 0.3 * gs
                    if combined > best_score and combined >= COMBINED_THRESHOLD:
                        best_score = combined
                        best_match = {
                            "registry_id": reg_id,
                            "overture_id": ov_id,
                            "match_score": round(combined / 100.0, 4),
                            "match_method": "name_geo",
                            "is_open": reg_is_open,
                        }

                if best_match:
                    matches.append(best_match)

            log.info("  processed %d / %d businesses, %d matches so far",
                     min(i + batch_size, len(reg_rows)), len(reg_rows), len(matches))

    log.info("  %s: %d matches from %d businesses (%.1f%% match rate)",
             city, len(matches), len(reg_rows),
             100 * len(matches) / len(reg_rows) if reg_rows else 0)
    return matches
