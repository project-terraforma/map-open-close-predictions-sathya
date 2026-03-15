"""
Build training data from Overture Maps changelogs.

The changelog tells us which businesses were REMOVED (likely closed) and
ADDED (likely open) between monthly releases. This gives hundreds of
thousands of labeled examples with full Overture metadata.

Usage:
    python -m src.step4_classifier.changelog_training build        # download + build parquet
    python -m src.step4_classifier.changelog_training build 50000  # limit sample size
    python -m src.step4_classifier.changelog_training stats        # show stats on existing file
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "changelog_training.parquet"

# Feb 2026 changelog — most recent available
CHANGELOG_BASE = "s3://overturemaps-us-west-2/changelog/2026-02-18.0/theme=places/type=place"


def _extract_field(struct, field):
    """Safely extract from a DuckDB struct/dict."""
    if struct is None:
        return None
    if isinstance(struct, dict):
        return struct.get(field)
    try:
        return getattr(struct, field, None)
    except Exception:
        return None


def build_training_data(max_per_class=50000):
    """Download removed + added places from Overture changelog and build training parquet."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2'")

    RELEASE_JAN = "s3://overturemaps-us-west-2/release/2026-01-21.0/theme=places/type=place/*"
    RELEASE_FEB = "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/*"

    dfs = []

    # REMOVED: get IDs from changelog, then pull metadata from Jan release (where they last existed)
    log.info("Downloading removed place IDs from changelog...")
    try:
        removed_ids = con.execute(f"""
            SELECT id FROM read_parquet('{CHANGELOG_BASE}/change_type=removed/*',
                hive_partitioning=true)
            USING SAMPLE {max_per_class}
        """).fetchdf()
        log.info("  Got %d removed IDs, fetching metadata from Jan release...", len(removed_ids))

        con.execute("CREATE TEMP TABLE removed_ids AS SELECT id FROM removed_ids")
        df_removed = con.execute(f"""
            SELECT p.id, p.names, p.categories, p.confidence, p.sources,
                   p.websites, p.phones, p.emails, p.socials, p.brand, p.addresses
            FROM read_parquet('{RELEASE_JAN}', hive_partitioning=true) p
            INNER JOIN removed_ids r ON p.id = r.id
        """).fetchdf()

        df_removed["label"] = 0
        df_removed["change_type"] = "removed"
        dfs.append(df_removed)
        log.info("  Got %d removed places with metadata", len(df_removed))
        con.execute("DROP TABLE removed_ids")
    except Exception as e:
        log.error("Failed to download removed: %s", e)

    # ADDED: get IDs from changelog, then pull metadata from Feb release (where they now exist)
    log.info("Downloading added place IDs from changelog...")
    try:
        added_ids = con.execute(f"""
            SELECT id FROM read_parquet('{CHANGELOG_BASE}/change_type=added/*',
                hive_partitioning=true)
            USING SAMPLE {max_per_class}
        """).fetchdf()
        log.info("  Got %d added IDs, fetching metadata from Feb release...", len(added_ids))

        con.execute("CREATE TEMP TABLE added_ids AS SELECT id FROM added_ids")
        df_added = con.execute(f"""
            SELECT p.id, p.names, p.categories, p.confidence, p.sources,
                   p.websites, p.phones, p.emails, p.socials, p.brand, p.addresses
            FROM read_parquet('{RELEASE_FEB}', hive_partitioning=true) p
            INNER JOIN added_ids a ON p.id = a.id
        """).fetchdf()

        df_added["label"] = 1
        df_added["change_type"] = "added"
        dfs.append(df_added)
        log.info("  Got %d added places with metadata", len(df_added))
        con.execute("DROP TABLE added_ids")
    except Exception as e:
        log.error("Failed to download added: %s", e)

    if not dfs:
        log.error("No data downloaded!")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Convert complex DuckDB types to JSON strings for compatibility with extract_features
    for col in ["names", "categories", "sources", "websites", "phones",
                "emails", "socials", "brand", "addresses"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(_to_json_str)

    # Extract website URL for alive checking
    combined["website"] = combined["websites"].apply(_extract_first_website)

    log.info("Combined: %d rows (%d open, %d closed)",
             len(combined),
             (combined["label"] == 1).sum(),
             (combined["label"] == 0).sum())

    combined.to_parquet(OUTPUT_FILE, index=False)
    log.info("Saved to %s", OUTPUT_FILE)


def _to_json_str(val):
    """Convert DuckDB struct/list to JSON string."""
    import json
    if val is None:
        return None
    if isinstance(val, str):
        return val
    try:
        return json.dumps(val)
    except (TypeError, ValueError):
        return str(val)


def _extract_first_website(websites_json):
    """Extract first website URL from websites JSON."""
    import json
    if not websites_json:
        return None
    try:
        if isinstance(websites_json, str):
            data = json.loads(websites_json)
        else:
            data = websites_json
        if isinstance(data, list) and len(data) > 0:
            w = data[0]
            if isinstance(w, dict):
                return w.get("value") or w.get("url")
            return str(w)
    except Exception:
        pass
    return None


def show_stats():
    """Show stats on existing changelog training data."""
    if not OUTPUT_FILE.exists():
        log.error("No changelog training file. Run 'build' first.")
        return

    df = pd.read_parquet(OUTPUT_FILE)
    log.info("Changelog training data: %s", OUTPUT_FILE)
    log.info("  Total rows: %d", len(df))
    log.info("  Open (added): %d (%.0f%%)", (df["label"] == 1).sum(),
             100 * (df["label"] == 1).mean())
    log.info("  Closed (removed): %d (%.0f%%)", (df["label"] == 0).sum(),
             100 * (df["label"] == 0).mean())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        build_training_data(max_per_class=limit)
    elif cmd == "stats":
        show_stats()
    else:
        print("Usage: python -m src.step4_classifier.changelog_training [build|stats] [limit]")
