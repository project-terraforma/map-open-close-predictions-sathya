"""
Step 6: Web/API scoring — check online presence of matched businesses.

Uses available APIs (Yelp, Foursquare, website reachability) to produce a
composite web presence score. Skips APIs whose keys are not configured.

Usage:
    python -m src.step6_web             # all matched cities
    python -m src.step6_web sf paris    # specific cities
"""

import logging
import sys

import requests
from sqlalchemy import text

from src.config import engine, YELP_API_KEY, FOURSQUARE_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "chi": "chicago", "chicago": "chicago",
    "paris": "paris", "sg": "singapore", "singapore": "singapore",
    "london": "london", "ldn": "london", "mumbai": "mumbai",
}


def _check_website(url: str) -> float:
    """Return 1.0 if website is reachable, 0.0 otherwise."""
    if not url:
        return 0.0
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        return 1.0 if resp.status_code < 400 else 0.0
    except Exception:
        return 0.0


def _yelp_score(name: str, lat: float, lon: float) -> float | None:
    """Search Yelp for the business, return normalized score or None if no key."""
    if not YELP_API_KEY:
        return None
    try:
        resp = requests.get("https://api.yelp.com/v3/businesses/search", params={
            "term": name, "latitude": lat, "longitude": lon, "limit": 1,
        }, headers={"Authorization": f"Bearer {YELP_API_KEY}"}, timeout=10)
        data = resp.json()
        businesses = data.get("businesses", [])
        if businesses:
            biz = businesses[0]
            if biz.get("is_closed", False):
                return 0.1
            review_count = biz.get("review_count", 0)
            return min(1.0, 0.3 + 0.7 * min(review_count / 50, 1.0))
        return 0.2
    except Exception:
        return None


def _foursquare_score(name: str, lat: float, lon: float) -> float | None:
    """Search Foursquare for the business."""
    if not FOURSQUARE_API_KEY:
        return None
    try:
        resp = requests.get("https://api.foursquare.com/v3/places/search", params={
            "query": name, "ll": f"{lat},{lon}", "limit": 1,
        }, headers={"Authorization": FOURSQUARE_API_KEY}, timeout=10)
        data = resp.json()
        results = data.get("results", [])
        if results:
            closed = results[0].get("closed_bucket", "unsure")
            if closed == "likely_closed":
                return 0.1
            return 0.8
        return 0.2
    except Exception:
        return None


def score_city(city: str):
    """Score all matched businesses for a city using web signals."""
    log.info("Web scoring for %s", city)

    available_apis = []
    if YELP_API_KEY:
        available_apis.append("yelp")
    if FOURSQUARE_API_KEY:
        available_apis.append("foursquare")
    available_apis.append("website_check")

    log.info("  Available APIs: %s", ", ".join(available_apis))

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT m.overture_id, o.name, o.latitude, o.longitude, o.website
            FROM overture.matched m
            JOIN registries.businesses r ON r.id = m.registry_id
            JOIN overture.places o ON o.id = m.overture_id
            WHERE r.city = :city
        """), {"city": city}).fetchall()

    log.info("  %d businesses to score", len(rows))

    results = []
    for i, (ov_id, name, lat, lon, website) in enumerate(rows):
        web_presence = _check_website(website)
        yelp = _yelp_score(name, lat, lon)
        fsq = _foursquare_score(name, lat, lon)

        scores = [web_presence]
        if yelp is not None:
            scores.append(yelp)
        if fsq is not None:
            scores.append(fsq)
        composite = sum(scores) / len(scores)

        results.append({
            "overture_id": ov_id,
            "city": city,
            "yelp_score": yelp,
            "foursquare_score": fsq,
            "web_presence_score": web_presence,
            "composite_web_score": composite,
        })

        if (i + 1) % 100 == 0:
            log.info("  scored %d / %d", i + 1, len(rows))

    log.info("Storing %d web scores…", len(results))
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM web_scores.results WHERE city = :city"), {"city": city})
        for r in results:
            conn.execute(text("""
                INSERT INTO web_scores.results
                    (overture_id, city, yelp_score, foursquare_score,
                     web_presence_score, composite_web_score)
                VALUES (:overture_id, :city, :yelp_score, :foursquare_score,
                        :web_presence_score, :composite_web_score)
            """), r)

    with engine.begin() as conn:
        for r in results:
            conn.execute(text("""
                UPDATE predictions.results
                SET composite_web_score = :score
                WHERE overture_id = :ov_id
            """), {"score": r["composite_web_score"], "ov_id": r["overture_id"]})

    log.info("Done web scoring for %s: avg composite=%.3f",
             city, sum(r["composite_web_score"] for r in results) / len(results) if results else 0)


def run(cities: list[str] | None = None):
    if cities:
        targets = [ALIASES.get(c, c) for c in cities]
    else:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT b.city FROM overture.matched m "
                "JOIN registries.businesses b ON b.id = m.registry_id"
            )).fetchall()
            targets = [r[0] for r in rows]

    for city in targets:
        score_city(city)


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
