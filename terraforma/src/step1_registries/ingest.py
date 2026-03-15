"""
Insert downloaded registry records into the database.
"""

import logging

from sqlalchemy import text

from src.config import engine

log = logging.getLogger(__name__)

INSERT_SQL = text("""
    INSERT INTO registries.businesses
        (city, name, name_normalized, address, address_normalized,
         latitude, longitude, category, is_open, raw_source)
    VALUES
        (:city, :name, :name_normalized, :address, :address_normalized,
         :latitude, :longitude, :category, :is_open, CAST(:raw_source AS jsonb))
""")

BATCH_SIZE = 1000


def ingest(records: list[dict], city: str):
    """Bulk-insert records for a city, clearing old data first."""
    log.info("Ingesting %d records for %s", len(records), city)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM registries.businesses WHERE city = :city"),
            {"city": city},
        )
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.execute(INSERT_SQL, batch)
            log.info("  inserted %d / %d", min(i + BATCH_SIZE, len(records)), len(records))

    log.info("Done ingesting %s", city)
