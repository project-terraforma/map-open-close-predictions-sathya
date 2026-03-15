"""
Build ground truth test sets for additional cities.

Closed: from changelog_training.parquet (removed businesses, filtered by US addresses)
Open: from our existing overture.places DB (high-confidence, stable businesses)

Usage:
    python -m src.step7_ground_truth.diff_ground_truth build
    python -m src.step7_ground_truth.diff_ground_truth show
"""

import ast
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "diff_ground_truth.parquet"
CHANGELOG_FILE = PROJECT_ROOT / "changelog_training.parquet"

# Cities we have in the DB
TEST_CITIES = ["san_francisco", "new_york", "chicago"]

TARGET_PER_CLASS = 25


def _parse_str(val):
    """Parse a string that might be JSON or Python repr."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except Exception:
        pass
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def _extract_name(names_val):
    if not names_val:
        return None
    if isinstance(names_val, str):
        names_val = _parse_str(names_val)
        if isinstance(names_val, str):
            return names_val
    if isinstance(names_val, dict):
        return names_val.get("primary", names_val.get("value", str(names_val)))
    if isinstance(names_val, list) and len(names_val) > 0:
        n = names_val[0]
        if isinstance(n, dict):
            return n.get("value", n.get("primary", str(n)))
        return str(n)
    return str(names_val)


def _extract_address(addr_val):
    if not addr_val:
        return None
    if isinstance(addr_val, str):
        addr_val = _parse_str(addr_val)
        if isinstance(addr_val, str):
            return addr_val
    if isinstance(addr_val, list) and len(addr_val) > 0:
        a = addr_val[0]
        if isinstance(a, dict):
            return a.get("freeform", a.get("value", str(a)))
        return str(a)
    if isinstance(addr_val, dict):
        return addr_val.get("freeform", addr_val.get("value", str(addr_val)))
    return str(addr_val)


def _extract_locality(addr_val):
    """Extract city/locality from address."""
    if not addr_val:
        return None
    if isinstance(addr_val, str):
        addr_val = _parse_str(addr_val)
        if isinstance(addr_val, str):
            return None
    if isinstance(addr_val, list) and len(addr_val) > 0:
        a = addr_val[0]
        if isinstance(a, dict):
            return (a.get("locality") or "").lower().strip()
    if isinstance(addr_val, dict):
        return (addr_val.get("locality") or "").lower().strip()
    return None


def _extract_country(addr_val):
    """Extract country from address."""
    if not addr_val:
        return None
    if isinstance(addr_val, str):
        addr_val = _parse_str(addr_val)
        if isinstance(addr_val, str):
            return None
    if isinstance(addr_val, list) and len(addr_val) > 0:
        a = addr_val[0]
        if isinstance(a, dict):
            return (a.get("country") or "").upper().strip()
    if isinstance(addr_val, dict):
        return (addr_val.get("country") or "").upper().strip()
    return None


# Map our city keys to locality strings in Overture addresses
CITY_LOCALITY_MAP = {
    "san_francisco": ["san francisco"],
    "new_york": ["new york", "brooklyn", "bronx", "queens", "manhattan"],
    "chicago": ["chicago"],
}


def build_ground_truth():
    """Build ground truth from changelog (closed) + DB (open)."""
    if not CHANGELOG_FILE.exists():
        log.error("No changelog_training.parquet. Run: python -m src.step4_classifier.changelog_training build")
        return

    log.info("Loading changelog data...")
    cl_df = pd.read_parquet(CHANGELOG_FILE)
    closed_all = cl_df[cl_df["label"] == 0].copy()
    log.info("  %d removed businesses available", len(closed_all))

    # Extract location info from addresses
    closed_all["name"] = closed_all["names"].apply(_extract_name)
    closed_all["address"] = closed_all["addresses"].apply(_extract_address)
    closed_all["locality"] = closed_all["addresses"].apply(_extract_locality)
    closed_all["country"] = closed_all["addresses"].apply(_extract_country)

    # Filter to US only
    us_closed = closed_all[closed_all["country"] == "US"]
    log.info("  %d US removed businesses", len(us_closed))

    from src.config import engine
    from sqlalchemy import text

    all_results = []

    for city in TEST_CITIES:
        log.info("\n=== %s ===", city.upper())

        # CLOSED: match by locality
        localities = CITY_LOCALITY_MAP.get(city, [])
        city_closed = us_closed[us_closed["locality"].isin(localities)]
        city_closed = city_closed[city_closed["name"].notna() & (city_closed["name"].str.len() > 2)]

        if len(city_closed) > TARGET_PER_CLASS:
            city_closed = city_closed.sample(n=TARGET_PER_CLASS, random_state=42)

        city_closed["label_gt"] = "CLOSED"
        city_closed["city"] = city
        all_results.append(city_closed[["id", "name", "address", "label_gt", "city"]])
        log.info("  Closed: %d businesses", len(city_closed))

        # OPEN: from DB
        try:
            with engine.connect() as conn:
                open_rows = pd.read_sql(text("""
                    SELECT id, name, address
                    FROM overture.places
                    WHERE city = :city
                    AND confidence > 0.8
                    AND website IS NOT NULL
                    AND name IS NOT NULL AND length(name) > 2
                    ORDER BY random()
                    LIMIT :limit
                """), conn, params={"city": city, "limit": TARGET_PER_CLASS})

            open_rows["label_gt"] = "OPEN"
            open_rows["city"] = city
            all_results.append(open_rows[["id", "name", "address", "label_gt", "city"]])
            log.info("  Open: %d businesses", len(open_rows))
        except Exception as e:
            log.warning("  DB error for %s: %s", city, e)

    if not all_results:
        log.error("No data collected!")
        return

    result = pd.concat(all_results, ignore_index=True)
    result.to_parquet(OUTPUT_FILE, index=False)
    log.info("\nSaved %d ground truth businesses to %s", len(result), OUTPUT_FILE)

    for city in TEST_CITIES:
        subset = result[result["city"] == city]
        n_open = (subset["label_gt"] == "OPEN").sum()
        n_closed = (subset["label_gt"] == "CLOSED").sum()
        log.info("  %s: %d open, %d closed", city, n_open, n_closed)


def show_ground_truth():
    if not OUTPUT_FILE.exists():
        log.error("No diff ground truth file. Run 'build' first.")
        return

    df = pd.read_parquet(OUTPUT_FILE)
    for city in df["city"].unique():
        subset = df[df["city"] == city].sort_values(["label_gt", "name"])
        print(f"\n=== {city.upper()} ===")
        print(f"# | Name | Address | Label")
        print(f"--|------|---------|------")
        for i, (_, r) in enumerate(subset.iterrows(), 1):
            print(f"{i} | {r['name']} | {r['address']} | {r['label_gt']}")
        n_open = (subset["label_gt"] == "OPEN").sum()
        n_closed = (subset["label_gt"] == "CLOSED").sum()
        print(f"Total: {n_open} open, {n_closed} closed")


def load_to_db():
    """Load diff ground truth into ground_truth.labels + ensure overture.places has the rows."""
    if not OUTPUT_FILE.exists():
        log.error("No diff ground truth parquet. Run 'build' first.")
        return

    from src.config import engine
    from sqlalchemy import text

    df = pd.read_parquet(OUTPUT_FILE)
    log.info("Loading %d ground truth rows into DB...", len(df))

    with engine.begin() as conn:
        # Ensure ground_truth schema exists
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ground_truth"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ground_truth.labels (
                overture_id TEXT PRIMARY KEY,
                city TEXT,
                google_place_id TEXT,
                business_status TEXT,
                is_open BOOLEAN
            )
        """))

        # Clear old ground truth labels for these cities
        for city in df["city"].unique():
            conn.execute(text(
                "DELETE FROM ground_truth.labels WHERE city = :city"
            ), {"city": city})
            log.info("  Cleared old ground truth for %s", city)

        loaded = 0

        for _, row in df.iterrows():
            ov_id = row["id"]
            is_open = row["label_gt"] == "OPEN"
            city = row["city"]
            biz_status = "OPERATIONAL" if is_open else "CLOSED_PERMANENTLY"

            # For OPEN businesses, they should already be in overture.places
            # For CLOSED businesses (from changelog), they won't be — check
            exists = conn.execute(text(
                "SELECT 1 FROM overture.places WHERE id = :oid"
            ), {"oid": ov_id}).fetchone()

            if not exists:
                # Insert a minimal row into overture.places so ensemble JOIN works
                conn.execute(text("""
                    INSERT INTO overture.places (id, name, address, city, confidence, latitude, longitude)
                    VALUES (:id, :name, :address, :city, 0.5, 0, 0)
                    ON CONFLICT (id) DO NOTHING
                """), {
                    "id": ov_id,
                    "name": row["name"],
                    "address": row.get("address"),
                    "city": city,
                })

            # Upsert into ground_truth.labels
            conn.execute(text(
                "DELETE FROM ground_truth.labels WHERE overture_id = :oid"
            ), {"oid": ov_id})
            conn.execute(text("""
                INSERT INTO ground_truth.labels
                    (overture_id, city, google_place_id, business_status, is_open)
                VALUES (:overture_id, :city, :gid, :status, :is_open)
            """), {
                "overture_id": ov_id,
                "city": city,
                "gid": "",
                "status": biz_status,
                "is_open": is_open,
            })
            loaded += 1

        log.info("Loaded %d ground truth labels", loaded)

    # Show summary
    with engine.connect() as conn:
        for city in df["city"].unique():
            result = conn.execute(text(
                "SELECT COUNT(*), SUM(CASE WHEN is_open THEN 1 ELSE 0 END) "
                "FROM ground_truth.labels WHERE city = :city"
            ), {"city": city}).fetchone()
            log.info("  %s: %d total (%d open, %d closed)",
                     city, result[0], result[1], result[0] - result[1])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_ground_truth()
    elif cmd == "show":
        show_ground_truth()
    elif cmd == "load":
        load_to_db()
    else:
        print("Usage: python -m src.step7_ground_truth.diff_ground_truth [build|show|load]")
