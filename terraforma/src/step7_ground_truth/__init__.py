"""
Step 7: Ground truth labeling using Google Places API.

Queries Google Places to get the real-time business_status for matched
businesses and stores results in ground_truth.labels.

Usage:
    python -m src.step7_ground_truth balanced sf     # hunt for 15 open + 15 closed (Google-confirmed)
    python -m src.step7_ground_truth balanced sf 20 20  # custom target
    python -m src.step7_ground_truth fresh sf 25 25     # fresh GT from random Overture places
"""

import logging
import random
import sys
import time

import requests
from sqlalchemy import text

from src.config import engine, GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "chi": "chicago", "chicago": "chicago",
}

GOOGLE_TEXTSEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def _find_place(name: str, lat: float, lon: float) -> dict | None:
    """Find a place using Google Places API (New) with cross-validation fields.

    Requests businessStatus + currentOpeningHours + userRatingCount + reviews
    so we can flag suspicious labels (e.g. OPERATIONAL but no hours, no reviews).
    """
    if not GOOGLE_API_KEY:
        log.error("GOOGLE_API_KEY not set")
        return None

    resp = requests.post(GOOGLE_TEXTSEARCH_URL, json={
        "textQuery": name,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 200.0,
            }
        },
        "maxResultCount": 1,
    }, headers={
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.businessStatus,"
            "places.formattedAddress,places.location,"
            "places.currentOpeningHours,places.userRatingCount,"
            "places.rating,places.reviews,places.websiteUri"
        ),
    }, timeout=10)

    data = resp.json()
    places = data.get("places", [])
    if not places:
        return None

    place = places[0]
    status = place.get("businessStatus", "UNKNOWN")
    has_hours = "currentOpeningHours" in place
    rating_count = place.get("userRatingCount", 0) or 0
    rating = place.get("rating")
    google_website = place.get("websiteUri")

    # Check for recent reviews (within last ~12 months)
    reviews = place.get("reviews", [])
    recent_reviews = 0
    for rev in reviews:
        pub_time = rev.get("publishTime", "")
        # Google returns ISO format like "2025-11-03T..."
        if pub_time >= "2025-03-01":
            recent_reviews += 1

    # Flag suspicious: API says OPERATIONAL but no hours, few ratings, no recent reviews
    suspicious = False
    if status == "OPERATIONAL" and not has_hours and rating_count < 5 and recent_reviews == 0:
        suspicious = True

    return {
        "google_place_id": place.get("id", ""),
        "business_status": status,
        "google_name": place.get("displayName", {}).get("text", ""),
        "google_address": place.get("formattedAddress", ""),
        "google_lat": place.get("location", {}).get("latitude"),
        "google_lon": place.get("location", {}).get("longitude"),
        "has_hours": has_hours,
        "rating_count": rating_count,
        "rating": rating,
        "recent_reviews": recent_reviews,
        "google_website": google_website,
        "suspicious": suspicious,
    }


def _store_label(conn, r: dict):
    """Upsert a single ground truth label."""
    conn.execute(text("DELETE FROM ground_truth.labels WHERE overture_id = :ov_id"),
                 {"ov_id": r["overture_id"]})
    conn.execute(text("""
        INSERT INTO ground_truth.labels
            (overture_id, city, google_place_id, business_status, is_open)
        VALUES (:overture_id, :city, :google_place_id, :business_status, :is_open)
    """), r)


def _get_candidates(city: str, registry_open: bool, limit: int, exclude_ids: set):
    """Get candidate businesses from matched data, excluding already-labeled ones."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (m.overture_id)
                   m.overture_id, o.name, o.latitude, o.longitude, m.is_open
            FROM overture.matched m
            JOIN overture.places o ON o.id = m.overture_id
            JOIN registries.businesses r ON r.id = m.registry_id
            WHERE r.city = :city AND m.is_open = :is_open
            ORDER BY m.overture_id, m.match_score DESC
            LIMIT :n
        """), {"city": city, "is_open": registry_open, "n": limit + len(exclude_ids)}).fetchall()

    # Filter out already-labeled and shuffle for variety
    candidates = [r for r in rows if r[0] not in exclude_ids]
    random.shuffle(candidates)
    return candidates


def balanced_ground_truth(city: str, target_open: int = 15, target_closed: int = 15):
    """Hunt for balanced ground truth: query Google until we have enough open AND closed.

    Strategy:
    - First, collect registry-closed candidates (more likely to be actually closed)
    - Query Google for each; keep going until we have target_closed confirmed closed
    - Then collect registry-open candidates for the open side
    - Also keep any Google-confirmed open businesses found while hunting for closed
    """
    key = ALIASES.get(city, city)
    log.info("=" * 60)
    log.info("BALANCED GROUND TRUTH: %s", key)
    log.info("Target: %d Google-confirmed open + %d Google-confirmed closed", target_open, target_closed)
    log.info("=" * 60)

    # Get already-labeled IDs to skip
    with engine.connect() as conn:
        existing = conn.execute(text(
            "SELECT overture_id, is_open FROM ground_truth.labels WHERE city = :city"
        ), {"city": key}).fetchall()

    existing_ids = {r[0] for r in existing}
    existing_open = sum(1 for r in existing if r[1])
    existing_closed = sum(1 for r in existing if not r[1])
    log.info("Already labeled: %d open, %d closed", existing_open, existing_closed)

    need_open = max(0, target_open - existing_open)
    need_closed = max(0, target_closed - existing_closed)
    log.info("Still need: %d open, %d closed", need_open, need_closed)

    if need_open == 0 and need_closed == 0:
        log.info("Already have enough! Nothing to do.")
        return

    found_open = 0
    found_closed = 0
    api_calls = 0
    stored_ids = set()

    # ── Phase 1: Hunt for CLOSED businesses ──
    # Query registry-closed candidates first (higher chance of being actually closed)
    # Also query registry-open (some will be closed on Google even if registry says open)
    if need_closed > 0:
        log.info("\n── Phase 1: Hunting for %d closed businesses ──", need_closed)

        # Get a large pool of registry-closed candidates
        closed_candidates = _get_candidates(key, False, 500, existing_ids | stored_ids)
        # Also try some registry-open candidates (Google may say closed)
        open_candidates = _get_candidates(key, True, 200, existing_ids | stored_ids)

        # Prioritize registry-closed, then sprinkle in open
        candidates = closed_candidates + open_candidates

        for ov_id, name, lat, lon, reg_open in candidates:
            if found_closed >= need_closed:
                break
            if ov_id in stored_ids:
                continue

            api_calls += 1
            log.info("  [API call %d] %s (registry=%s)",
                     api_calls, name, "open" if reg_open else "closed")

            place = _find_place(name, lat, lon)
            if not place:
                log.warning("    → Not found on Google")
                time.sleep(0.2)
                continue

            status = place["business_status"]
            is_open = status == "OPERATIONAL"
            log.info("    → Google: %s", status)

            r = {
                "overture_id": ov_id,
                "city": key,
                "google_place_id": place["google_place_id"],
                "business_status": status,
                "is_open": is_open,
            }

            # Store immediately
            with engine.begin() as conn:
                _store_label(conn, r)
            stored_ids.add(ov_id)

            if is_open:
                found_open += 1
            else:
                found_closed += 1
                log.info("    ✓ CLOSED #%d/%d collected", found_closed, need_closed)

            time.sleep(0.2)

    # Update needs (some open may have been found during closed hunting)
    need_open = max(0, need_open - found_open)

    # ── Phase 2: Collect OPEN businesses ──
    if need_open > 0:
        log.info("\n── Phase 2: Collecting %d open businesses ──", need_open)

        open_candidates = _get_candidates(key, True, need_open + 50, existing_ids | stored_ids)

        for ov_id, name, lat, lon, reg_open in open_candidates:
            if found_open >= (target_open - existing_open):
                break
            if ov_id in stored_ids:
                continue

            api_calls += 1
            log.info("  [API call %d] %s", api_calls, name)

            place = _find_place(name, lat, lon)
            if not place:
                log.warning("    → Not found on Google")
                time.sleep(0.2)
                continue

            status = place["business_status"]
            is_open = status == "OPERATIONAL"
            log.info("    → Google: %s", status)

            r = {
                "overture_id": ov_id,
                "city": key,
                "google_place_id": place["google_place_id"],
                "business_status": status,
                "is_open": is_open,
            }

            with engine.begin() as conn:
                _store_label(conn, r)
            stored_ids.add(ov_id)

            if is_open:
                found_open += 1
            else:
                found_closed += 1  # bonus closed

            time.sleep(0.2)

    # ── Summary ──
    with engine.connect() as conn:
        final = conn.execute(text("""
            SELECT sum(case when is_open then 1 else 0 end),
                   sum(case when not is_open then 1 else 0 end),
                   count(*)
            FROM ground_truth.labels WHERE city = :city
        """), {"city": key}).fetchone()

    log.info("\n" + "=" * 60)
    log.info("FINAL GROUND TRUTH FOR %s", key.upper())
    log.info("  Google-confirmed open:   %d", final[0])
    log.info("  Google-confirmed closed: %d", final[1])
    log.info("  Total labels:            %d", final[2])
    log.info("  API calls used:          %d", api_calls)
    log.info("=" * 60)


def fresh_ground_truth(city: str, target_open: int = 25, target_closed: int = 25):
    """Build ground truth from RANDOM Overture places — no structural gap.

    Unlike balanced_ground_truth (which pulls from overture.matched → registry),
    this samples from ALL overture.places in the city bbox. This means both
    open AND closed businesses have the same full Overture metadata.

    Cross-validates Google businessStatus with hours/reviews/ratings to flag
    suspicious labels you should spot-check.
    """
    key = ALIASES.get(city, city)

    # Get city bbox from config
    from src.config import CITY_BBOXES
    bbox = CITY_BBOXES.get(key)
    if not bbox:
        log.error("No bbox for city %s", key)
        return

    log.info("=" * 60)
    log.info("FRESH GROUND TRUTH: %s (from random Overture places)", key)
    log.info("Target: %d open + %d closed (Google-confirmed)", target_open, target_closed)
    log.info("=" * 60)

    # Clear old ground truth for this city so we start clean
    with engine.begin() as conn:
        deleted = conn.execute(text(
            "DELETE FROM ground_truth.labels WHERE city = :city"
        ), {"city": key}).rowcount
    if deleted:
        log.info("Cleared %d old labels for %s", deleted, key)

    # Categories to skip — solo practitioners and niche businesses with
    # little web presence. These are nearly impossible to verify via web crawling.
    SKIP_CATEGORIES = [
        "doctor", "dentist", "lawyer", "attorney", "chiropract",
        "optician", "optometrist", "therapist", "psycholog",
        "psychiatr", "counselor", "notary", "accountant", "tax_prep",
        "insurance_agent", "real_estate_agent", "tutor", "driving_school",
        "funeral", "cemetery",
    ]
    skip_pattern = "|".join(SKIP_CATEGORIES)

    # Get a large random sample from overture.places in this city
    # These ALL have full metadata (confidence, sources, websites, etc.)
    # Skip solo practitioners / hard-to-verify categories
    with engine.connect() as conn:
        candidates = conn.execute(text(f"""
            SELECT id, name, latitude, longitude, website, category
            FROM overture.places
            WHERE latitude BETWEEN :min_lat AND :max_lat
              AND longitude BETWEEN :min_lon AND :max_lon
              AND name IS NOT NULL
              AND name != ''
              AND (category IS NULL OR NOT category ~* :skip_pattern)
            ORDER BY random()
            LIMIT :n
        """), {
            "min_lat": bbox["min_lat"], "max_lat": bbox["max_lat"],
            "min_lon": bbox["min_lon"], "max_lon": bbox["max_lon"],
            "skip_pattern": skip_pattern,
            "n": (target_open + target_closed) * 8,  # oversample
        }).fetchall()

    log.info("Candidate pool: %d random Overture places", len(candidates))

    found_open = 0
    found_closed = 0
    api_calls = 0
    suspicious_count = 0

    for ov_id, name, lat, lon, website, category in candidates:
        if found_open >= target_open and found_closed >= target_closed:
            break

        api_calls += 1
        place = _find_place(name, lat, lon)
        if not place:
            log.info("  [%d] %s → NOT FOUND on Google", api_calls, name)
            time.sleep(0.2)
            continue

        status = place["business_status"]

        # Only accept definitive labels — skip UNKNOWN (not found on Google)
        if status == "OPERATIONAL":
            is_open = True
        elif status == "CLOSED_PERMANENTLY":
            is_open = False
        elif status == "CLOSED_TEMPORARILY":
            log.info("  [%d] %s → TEMPORARILY CLOSED (skipping)", api_calls, name)
            time.sleep(0.2)
            continue
        else:
            # UNKNOWN = Google couldn't find it. NOT evidence of closure, skip.
            log.info("  [%d] %s → %s (skipping — not a real business match)", api_calls, name, status)
            time.sleep(0.2)
            continue

        # Skip if we already have enough of this class
        if is_open and found_open >= target_open:
            time.sleep(0.2)
            continue
        if not is_open and found_closed >= target_closed:
            time.sleep(0.2)
            continue

        # Also skip entries that look like addresses/landmarks, not businesses
        # (e.g. "18TH STREET", "135 Main Street", "Pennsylvania")
        google_name = place["google_name"]
        if google_name and google_name.lower() != name.lower():
            # Google matched a different place — name mismatch, skip
            name_words = set(name.lower().split())
            google_words = set(google_name.lower().split())
            overlap = len(name_words & google_words) / max(len(name_words), 1)
            if overlap < 0.3:
                log.info("  [%d] %s → name mismatch (google='%s'), skipping",
                         api_calls, name, google_name)
                time.sleep(0.2)
                continue

        # Log with cross-validation info
        flag = ""
        if place["suspicious"]:
            flag = " *** SUSPICIOUS ***"
            suspicious_count += 1

        log.info("  [%d] %s → %s (hrs=%s, ratings=%d, recent_rev=%d, goog_name=%s)%s",
                 api_calls, name, status,
                 place["has_hours"], place["rating_count"],
                 place["recent_reviews"], google_name, flag)

        r = {
            "overture_id": ov_id,
            "city": key,
            "google_place_id": place["google_place_id"],
            "business_status": status,
            "is_open": is_open,
        }
        with engine.begin() as conn:
            _store_label(conn, r)

        if is_open:
            found_open += 1
        else:
            found_closed += 1
            log.info("    CLOSED #%d/%d", found_closed, target_closed)

        time.sleep(0.2)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("FRESH GROUND TRUTH FOR %s", key.upper())
    log.info("  Google-confirmed open:   %d", found_open)
    log.info("  Google-confirmed closed: %d", found_closed)
    log.info("  Total API calls:         %d", api_calls)
    log.info("  Suspicious labels:       %d (spot-check these!)", suspicious_count)
    log.info("=" * 60)

    if found_closed < target_closed:
        log.warning("Only found %d/%d closed — most businesses in Overture are open!",
                    found_closed, target_closed)
        log.info("Tip: increase the candidate pool or try a different city.")


def run(args: list[str] | None = None):
    if not args:
        print("Usage:")
        print("  python -m src.step7_ground_truth balanced sf       # 15 open + 15 closed (from matched)")
        print("  python -m src.step7_ground_truth balanced sf 20 20 # custom target")
        print("  python -m src.step7_ground_truth fresh sf 25 25    # from random Overture places (no structural gap)")
        return

    cmd = args[0]
    if cmd == "fresh":
        city = args[1] if len(args) > 1 else "sf"
        n_open = int(args[2]) if len(args) > 2 else 25
        n_closed = int(args[3]) if len(args) > 3 else 25
        fresh_ground_truth(city, n_open, n_closed)
    elif cmd == "balanced":
        city = args[1] if len(args) > 1 else "sf"
        n_open = int(args[2]) if len(args) > 2 else 15
        n_closed = int(args[3]) if len(args) > 3 else 15
        balanced_ground_truth(city, n_open, n_closed)
    else:
        # Legacy: treat as balanced with defaults
        city = args[0]
        n_open = int(args[1]) if len(args) > 1 else 15
        n_closed = int(args[2]) if len(args) > 2 else 15
        balanced_ground_truth(city, n_open, n_closed)


if __name__ == "__main__":
    args = sys.argv[1:]
    run(args)
