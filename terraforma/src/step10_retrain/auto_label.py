"""
Auto-labeling pipeline: use web crawling (Brave + Groq) to label businesses.

Only keeps high-confidence labels where Groq AI is >= 80% confident.
Skips OSM to minimize API calls — 1 Brave + 1 Groq per business.
The model is excluded to avoid circular training.
"""

import logging
import time

from sqlalchemy import text

from src.config import engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Confidence gates — based on web score (0-1)
OPEN_GATE = 0.75    # web score >= this → label as OPEN
CLOSED_GATE = 0.25  # web score <= this → label as CLOSED
LABEL_TTL_DAYS = 90

CITY_MAP = {
    "sf": ("san_francisco", "San Francisco"),
    "nyc": ("new_york", "New York City"),
    "chicago": ("chicago", "Chicago"),
}


def _ensure_table():
    """Create auto_labels table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions.auto_labels (
                overture_id     TEXT PRIMARY KEY,
                city            TEXT NOT NULL,
                osm_score       REAL,
                web_score       REAL,
                combined_score  REAL,
                auto_label      BOOLEAN NOT NULL,
                confidence      REAL NOT NULL,
                labeled_at      TIMESTAMP DEFAULT NOW(),
                ttl_expires_at  TIMESTAMP DEFAULT NOW() + INTERVAL '90 days',
                verified        BOOLEAN DEFAULT FALSE
            )
        """))


def auto_label(city_key: str, count: int = 100):
    """Auto-label `count` businesses in a city using web crawling only (1 Brave + 1 Groq per business)."""
    from src.step8_ensemble import web_crawl_signal

    if city_key not in CITY_MAP:
        log.error("Unknown city: %s. Use: %s", city_key, list(CITY_MAP.keys()))
        return

    db_city, city_label = CITY_MAP[city_key]
    _ensure_table()

    # Get candidates: matched businesses not already labeled
    with engine.connect() as conn:
        candidates = conn.execute(text("""
            SELECT o.id, o.name, o.latitude, o.longitude, o.website
            FROM overture.matched m
            JOIN overture.places o ON o.id = m.overture_id
            WHERE o.city = :city
            AND o.id NOT IN (SELECT overture_id FROM predictions.auto_labels)
            AND o.id NOT IN (SELECT overture_id FROM ground_truth.labels WHERE city = :city)
            AND o.name IS NOT NULL AND length(o.name) > 2
            ORDER BY random()
            LIMIT :lim
        """), {"city": db_city, "lim": count * 3}).fetchall()
        # Fetch 3x because confidence gate will reject many

    log.info("=" * 60)
    log.info("AUTO-LABELING: %s (%s)", db_city, city_label)
    log.info("Candidates: %d | Target: %d high-confidence labels", len(candidates), count)
    log.info("Gates: OPEN >= %.2f, CLOSED <= %.2f", OPEN_GATE, CLOSED_GATE)
    log.info("Method: 1 Brave + 1 Groq per business (no OSM)")
    log.info("=" * 60)

    labeled = 0
    skipped = 0

    for i, (ov_id, name, lat, lon, website) in enumerate(candidates):
        if labeled >= count:
            break

        log.info("\n[%d] %s", i + 1, name)

        try:
            w = web_crawl_signal(name, city_label, website)
            web_s = w.get("web_score") or 0.5

            log.info("  Web: %.2f (alive=%s, AI=%s %d%%)",
                     web_s, w.get("web_alive"), w.get("ai_judgment", "?"),
                     w.get("ai_confidence", 0))

            # Apply confidence gate
            if web_s >= OPEN_GATE:
                label = True  # OPEN
                conf = web_s
                log.info("  → LABELED OPEN (%.0f%%)", web_s * 100)
            elif web_s <= CLOSED_GATE:
                label = False  # CLOSED
                conf = 1 - web_s
                log.info("  → LABELED CLOSED (%.0f%%)", web_s * 100)
            else:
                skipped += 1
                log.info("  → SKIPPED (uncertain: %.0f%%)", web_s * 100)
                time.sleep(0.3)
                continue

            # Store
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO predictions.auto_labels
                        (overture_id, city, web_score, combined_score,
                         auto_label, confidence, labeled_at, ttl_expires_at)
                    VALUES (:oid, :city, :web, :web,
                            :label, :conf, NOW(), NOW() + INTERVAL '90 days')
                    ON CONFLICT (overture_id) DO UPDATE SET
                        web_score = :web, combined_score = :web,
                        auto_label = :label, confidence = :conf,
                        labeled_at = NOW(), ttl_expires_at = NOW() + INTERVAL '90 days'
                """), {"oid": ov_id, "city": db_city, "web": web_s,
                       "label": label, "conf": conf})

            labeled += 1
            time.sleep(0.8)

        except Exception as e:
            log.warning("  Error scoring %s: %s", name, e)
            continue

    log.info("\n" + "=" * 60)
    log.info("RESULTS: %d labeled, %d skipped (uncertain)", labeled, skipped)
    log.info("=" * 60)

    # Show distribution
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT auto_label, count(*) FROM predictions.auto_labels
            WHERE city = :city GROUP BY auto_label
        """), {"city": db_city}).fetchall()
    for row in stats:
        lbl = "OPEN" if row[0] else "CLOSED"
        log.info("  %s: %d", lbl, row[1])


def refresh_stale(city_key: str, batch_size: int = 20):
    """Re-verify the oldest expired labels."""
    from src.step8_ensemble import osm_signal, web_crawl_signal

    if city_key not in CITY_MAP:
        log.error("Unknown city: %s", city_key)
        return

    db_city, city_label = CITY_MAP[city_key]

    with engine.connect() as conn:
        expired = conn.execute(text("""
            SELECT a.overture_id, o.name, o.latitude, o.longitude, o.website
            FROM predictions.auto_labels a
            JOIN overture.places o ON o.id = a.overture_id
            WHERE a.city = :city AND a.ttl_expires_at < NOW()
            ORDER BY a.labeled_at ASC
            LIMIT :lim
        """), {"city": db_city, "lim": batch_size}).fetchall()

    if not expired:
        log.info("No expired labels for %s", db_city)
        return

    log.info("Refreshing %d expired labels for %s...", len(expired), db_city)

    for ov_id, name, lat, lon, website in expired:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                o = pool.submit(osm_signal, name, lat, lon).result()
                w = pool.submit(web_crawl_signal, name, city_label, website).result()

            osm_s = o.get("osm_score") or 0.5
            web_s = w.get("web_score") or 0.5
            combined = (W_OSM * osm_s) + (W_WEB * web_s)

            if combined >= OPEN_GATE or combined <= CLOSED_GATE:
                label = combined >= OPEN_GATE
                conf = combined if label else (1 - combined)
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE predictions.auto_labels
                        SET osm_score = :osm, web_score = :web, combined_score = :combined,
                            auto_label = :label, confidence = :conf,
                            labeled_at = NOW(), ttl_expires_at = NOW() + INTERVAL '90 days',
                            verified = TRUE
                        WHERE overture_id = :oid
                    """), {"oid": ov_id, "osm": osm_s, "web": web_s,
                           "combined": combined, "label": label, "conf": conf})
                log.info("  %s: refreshed → %s (%.1f%%)", name,
                         "OPEN" if label else "CLOSED", combined * 100)
            else:
                # Still uncertain — remove it
                with engine.begin() as conn:
                    conn.execute(text(
                        "DELETE FROM predictions.auto_labels WHERE overture_id = :oid"
                    ), {"oid": ov_id})
                log.info("  %s: removed (now uncertain)", name)

            time.sleep(1.0)
        except Exception as e:
            log.warning("  Error refreshing %s: %s", name, e)
