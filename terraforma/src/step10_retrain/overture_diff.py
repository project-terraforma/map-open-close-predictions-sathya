"""
Overture Monthly Diff: compare current DB snapshot against latest S3 release.

Since Overture only keeps the latest release on S3, we use our DB (loaded from
a previous release) as the "old" snapshot and query S3 for the "new" one.

Business in DB but GONE from latest release → likely CLOSED
Business in BOTH → likely OPEN
Business that LOST confidence/sources → weakened (possible closure signal)

Usage:
    python -m src.step10_retrain diff sf
    python -m src.step10_retrain diff all
"""

import logging

import duckdb
import pandas as pd
from sqlalchemy import text

from src.config import engine, CITY_BBOXES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Update this when a new Overture release drops
LATEST_RELEASE = "2026-02-18.0"
OVERTURE_PATH = f"s3://overturemaps-us-west-2/release/{LATEST_RELEASE}/theme=places/type=place/*"

QUERY_TEMPLATE = """
    SELECT
        id,
        names.primary               AS name,
        categories.primary          AS category,
        confidence
    FROM read_parquet('{path}', filename=true, hive_partitioning=1)
    WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
      AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
      AND names.primary IS NOT NULL
"""


def _get_duckdb():
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    return con


def _get_db_snapshot(city: str) -> set:
    """Get all Overture IDs we currently have in our DB for this city."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, confidence FROM overture.places WHERE city = :city"
        ), {"city": city}).fetchall()
    return {r[0]: r[1] for r in rows}


def _query_latest(city: str) -> pd.DataFrame:
    """Query the latest Overture release from S3."""
    bbox = CITY_BBOXES[city]
    con = _get_duckdb()
    sql = QUERY_TEMPLATE.format(path=OVERTURE_PATH, **bbox)
    log.info("  Querying latest release (%s) from S3...", LATEST_RELEASE)
    df = con.execute(sql).fetchdf()
    con.close()
    log.info("  → %d POIs from S3", len(df))
    return df


def store_diff_labels(city: str, max_per_class: int = 500):
    """Compare DB snapshot vs latest S3 release and store diff labels."""
    if city not in CITY_BBOXES:
        log.error("Unknown city: %s", city)
        return

    log.info("=" * 60)
    log.info("OVERTURE DIFF: %s", city)
    log.info("  DB snapshot vs S3 release %s", LATEST_RELEASE)
    log.info("=" * 60)

    # Get our DB snapshot (the "old" data)
    db_data = _get_db_snapshot(city)
    db_ids = set(db_data.keys())
    log.info("  DB snapshot: %d POIs", len(db_ids))

    # Get latest from S3 (the "new" data)
    s3_df = _query_latest(city)
    s3_ids = set(s3_df["id"])

    # Diff
    disappeared = db_ids - s3_ids       # In DB but gone from S3 → CLOSED
    still_present = db_ids & s3_ids     # In both → OPEN
    new_additions = s3_ids - db_ids     # New in S3, not in DB → ignore

    log.info("\nDiff results:")
    log.info("  Disappeared: %d (→ closed labels)", len(disappeared))
    log.info("  Still present: %d (→ open labels)", len(still_present))
    log.info("  New in S3: %d (ignored)", len(new_additions))

    # Check confidence drops for still-present businesses
    s3_conf = s3_df.set_index("id")["confidence"]
    conf_dropped = []
    for oid in still_present:
        old_conf = db_data.get(oid, 0) or 0
        new_conf = s3_conf.get(oid, 0) or 0
        if old_conf - new_conf > 0.15:  # Confidence dropped by 15%+
            conf_dropped.append(oid)

    if conf_dropped:
        log.info("  Confidence dropped 15%%+: %d (→ weak closed labels)", len(conf_dropped))

    # Build label sets
    closed_ids = list(disappeared) + conf_dropped
    open_ids = [oid for oid in still_present if oid not in conf_dropped]

    # Cap and balance
    import random
    random.seed(42)
    if len(closed_ids) > max_per_class:
        closed_ids = random.sample(closed_ids, max_per_class)
    if len(open_ids) > max_per_class:
        open_ids = random.sample(open_ids, max_per_class)

    # Balance open to roughly match closed (± 20%)
    target_open = min(len(open_ids), int(len(closed_ids) * 1.2))
    if target_open > 0 and len(open_ids) > target_open:
        open_ids = random.sample(open_ids, target_open)

    log.info("\nBalanced labels: %d closed, %d open", len(closed_ids), len(open_ids))

    if len(closed_ids) == 0 and len(open_ids) == 0:
        log.info("No diff labels to store (DB and S3 are identical).")
        log.info("This is expected if the DB was loaded from the same release (%s).", LATEST_RELEASE)
        log.info("\nTo generate diff labels, wait for the next Overture release,")
        log.info("update LATEST_RELEASE in overture_diff.py, and re-run.")
        return

    # Ensure table
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

    # Store
    stored = 0
    with engine.begin() as conn:
        for oid in closed_ids:
            conf = 0.85 if oid in disappeared else 0.65
            try:
                conn.execute(text("""
                    INSERT INTO predictions.auto_labels
                        (overture_id, city, auto_label, confidence, labeled_at,
                         ttl_expires_at)
                    VALUES (:oid, :city, FALSE, :conf, NOW(),
                            NOW() + INTERVAL '180 days')
                    ON CONFLICT (overture_id) DO NOTHING
                """), {"oid": oid, "city": city, "conf": conf})
                stored += 1
            except Exception:
                continue

        for oid in open_ids:
            try:
                conn.execute(text("""
                    INSERT INTO predictions.auto_labels
                        (overture_id, city, auto_label, confidence, labeled_at,
                         ttl_expires_at)
                    VALUES (:oid, :city, TRUE, 0.80, NOW(),
                            NOW() + INTERVAL '180 days')
                    ON CONFLICT (overture_id) DO NOTHING
                """), {"oid": oid, "city": city})
                stored += 1
            except Exception:
                continue

    log.info("\nStored %d diff-based auto-labels for %s", stored, city)

    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT auto_label, count(*) FROM predictions.auto_labels
            WHERE city = :city GROUP BY auto_label ORDER BY auto_label
        """), {"city": city}).fetchall()
    for lbl, cnt in stats:
        log.info("  %s: %d", "OPEN" if lbl else "CLOSED", cnt)
