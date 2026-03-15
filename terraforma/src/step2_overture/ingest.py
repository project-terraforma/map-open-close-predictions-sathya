"""
Insert Overture places into the database.
"""

import logging

from sqlalchemy import text

from src.config import engine

log = logging.getLogger(__name__)

INSERT_SQL = text("""
    INSERT INTO overture.places
        (id, city, name, name_normalized, category, confidence,
         latitude, longitude, address, phone, website, raw_json)
    VALUES
        (:id, :city, :name, :name_normalized, :category, :confidence,
         :latitude, :longitude, :address, :phone, :website, CAST(:raw_json AS jsonb))
    ON CONFLICT (id) DO NOTHING
""")

BATCH_SIZE = 1000


def ingest(records: list[dict], city: str):
    """Bulk-insert Overture places for a city, clearing old data first."""
    log.info("Ingesting %d Overture places for %s", len(records), city)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM overture.places WHERE city = :city"),
            {"city": city},
        )
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            conn.execute(INSERT_SQL, batch)
            log.info("  inserted %d / %d", min(i + BATCH_SIZE, len(records)), len(records))

    log.info("Done ingesting Overture places for %s", city)
