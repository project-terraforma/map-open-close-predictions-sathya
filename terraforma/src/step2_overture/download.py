"""
Query Overture Maps places via DuckDB (reads GeoParquet from S3, no auth needed).
"""

import json
import logging

import duckdb

from src.config import CITY_BBOXES
from src.step1_registries.normalize import normalize_name, normalize_address

log = logging.getLogger(__name__)

OVERTURE_RELEASE = "2026-02-18.0"
OVERTURE_PATH = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=places/type=place/*"

QUERY = """
    SELECT
        id,
        names.primary               AS name,
        categories.primary          AS category,
        confidence,
        ST_Y(geometry)              AS latitude,
        ST_X(geometry)              AS longitude,
        addresses[1].freeform       AS address,
        phones[1]                   AS phone,
        websites[1]                 AS website,
        sources                     AS sources
    FROM read_parquet('{path}', filename=true, hive_partitioning=1)
    WHERE bbox.xmin BETWEEN {min_lon} AND {max_lon}
      AND bbox.ymin BETWEEN {min_lat} AND {max_lat}
"""


def _get_connection():
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    return con


def download_city(city: str) -> list[dict]:
    bbox = CITY_BBOXES.get(city)
    if not bbox:
        raise ValueError(f"No bounding box for city: {city}")

    log.info("Querying Overture Maps for %s (release %s) …", city, OVERTURE_RELEASE)
    con = _get_connection()

    sql = QUERY.format(path=OVERTURE_PATH, **bbox)
    rows = con.execute(sql).fetchall()
    columns = ["id", "name", "category", "confidence", "latitude", "longitude",
               "address", "phone", "website", "sources"]

    log.info("  got %d raw places for %s", len(rows), city)

    records = []
    for row in rows:
        raw = dict(zip(columns, row))
        name = raw["name"]
        if not name:
            continue
        # Build raw_json from sources (convert non-serializable types)
        raw_json = {k: str(v) if v is not None else None for k, v in raw.items()}
        records.append({
            "id": raw["id"],
            "city": city,
            "name": name,
            "name_normalized": normalize_name(name),
            "category": raw["category"],
            "confidence": raw["confidence"],
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "address": raw["address"],
            "phone": raw["phone"],
            "website": raw["website"],
            "raw_json": json.dumps(raw_json),
        })

    log.info("  %d valid places for %s", len(records), city)
    con.close()
    return records
