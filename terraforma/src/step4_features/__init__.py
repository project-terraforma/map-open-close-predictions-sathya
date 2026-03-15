"""
Step 4: Feature engineering for the open/closed classifier.

Extracts features from matched registry + Overture data:
- Match quality features (score, method)
- Overture metadata features (confidence, has phone, has website, category)
- Spatial features (neighbor density, nearby open/closed ratio)

Usage:
    python -m src.step4_features           # all matched cities
    python -m src.step4_features sf        # just San Francisco
"""

import logging
import sys

import pandas as pd
from sqlalchemy import text

from src.config import engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "paris": "paris",
    "sg": "singapore", "singapore": "singapore",
    "london": "london", "ldn": "london",
    "mumbai": "mumbai",
}


def extract_features(cities: list[str] | None = None) -> pd.DataFrame:
    """Build feature matrix from matched businesses."""

    query = """
        SELECT
            m.id           AS match_id,
            m.registry_id,
            m.overture_id,
            m.match_score,
            m.is_open,
            r.city,
            r.name_normalized  AS reg_name,
            r.address          AS reg_address,
            o.name_normalized  AS ov_name,
            o.category,
            o.confidence       AS ov_confidence,
            o.phone,
            o.website,
            o.latitude,
            o.longitude
        FROM overture.matched m
        JOIN registries.businesses r ON r.id = m.registry_id
        JOIN overture.places o       ON o.id = m.overture_id
    """

    if cities:
        keys = [ALIASES.get(c, c) for c in cities]
        placeholders = ", ".join(f"'{k}'" for k in keys)
        query += f" WHERE r.city IN ({placeholders})"

    log.info("Loading matched data…")
    df = pd.read_sql(query, engine)
    log.info("  %d matched records loaded", len(df))

    if df.empty:
        log.warning("No matched data found")
        return df

    # ── Feature extraction ───────────────────────────────────────────

    # Match quality
    df["feat_match_score"] = df["match_score"].fillna(0)

    # Overture confidence
    df["feat_ov_confidence"] = df["ov_confidence"].fillna(0)

    # Has contact info (signals active business)
    df["feat_has_phone"] = df["phone"].notna().astype(int)
    df["feat_has_website"] = df["website"].notna().astype(int)
    df["feat_has_address"] = df["reg_address"].notna().astype(int)

    # Category encoding — top categories get their own feature, rest grouped
    top_categories = df["category"].value_counts().head(20).index.tolist()
    df["feat_category"] = df["category"].where(df["category"].isin(top_categories), "other")
    df["feat_category"] = df["feat_category"].fillna("unknown")

    # Name length (longer names = more specific = easier to verify)
    df["feat_name_length"] = df["ov_name"].fillna("").str.len()

    # Neighbor density: count of other matched places within ~200m (using df itself)
    log.info("Computing neighbor density…")
    from scipy.spatial import cKDTree
    import numpy as np

    df["feat_neighbor_density"] = 0
    for city in df["city"].unique():
        mask = df["city"] == city
        coords = df.loc[mask, ["latitude", "longitude"]].values
        if len(coords) < 2:
            continue
        # Approximate: convert degrees to meters (rough)
        coords_m = coords.copy()
        coords_m[:, 0] *= 111_000  # lat to meters
        coords_m[:, 1] *= 111_000 * np.cos(np.radians(coords[:, 0].mean()))  # lon to meters
        tree = cKDTree(coords_m)
        counts = tree.query_ball_point(coords_m, r=200, return_length=True)
        df.loc[mask, "feat_neighbor_density"] = np.array(counts) - 1  # exclude self

    # City encoding
    df["feat_city"] = df["city"]

    log.info("Feature extraction complete: %d records, %d features",
             len(df), len([c for c in df.columns if c.startswith("feat_")]))

    return df


FEATURE_COLS = [
    "feat_match_score",
    "feat_ov_confidence",
    "feat_has_phone",
    "feat_has_website",
    "feat_has_address",
    "feat_name_length",
    "feat_neighbor_density",
]

CATEGORICAL_COLS = ["feat_category", "feat_city"]
TARGET_COL = "is_open"


def run(cities: list[str] | None = None):
    df = extract_features(cities)
    if df.empty:
        return

    # Print summary
    n_open = df["is_open"].sum()
    n_closed = len(df) - n_open
    log.info("Label distribution: %d open, %d closed (%.1f%% open)",
             n_open, n_closed, 100 * n_open / len(df))

    for col in FEATURE_COLS + CATEGORICAL_COLS:
        if col in df.columns:
            log.info("  %s: mean=%.2f" if df[col].dtype != object else "  %s: %d unique",
                     col, df[col].mean() if df[col].dtype != object else df[col].nunique())

    return df


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
