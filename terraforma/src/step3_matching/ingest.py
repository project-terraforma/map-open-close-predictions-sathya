"""
Insert matches into overture.matched table.
"""

import logging

from sqlalchemy import text

from src.config import engine

log = logging.getLogger(__name__)

INSERT_SQL = text("""
    INSERT INTO overture.matched
        (registry_id, overture_id, match_score, match_method, is_open)
    VALUES
        (:registry_id, :overture_id, :match_score, :match_method, :is_open)
""")

BATCH_SIZE = 1000


def ingest(matches: list[dict], city: str):
    """Bulk-insert match results, clearing old matches for this city first."""
    log.info("Ingesting %d matches for %s", len(matches), city)

    with engine.begin() as conn:
        # Delete old matches for this city via registry_id
        conn.execute(text("""
            DELETE FROM overture.matched
            WHERE registry_id IN (
                SELECT id FROM registries.businesses WHERE city = :city
            )
        """), {"city": city})

        for i in range(0, len(matches), BATCH_SIZE):
            batch = matches[i : i + BATCH_SIZE]
            conn.execute(INSERT_SQL, batch)
            log.info("  inserted %d / %d", min(i + BATCH_SIZE, len(matches)), len(matches))

    log.info("Done ingesting matches for %s", city)
