-- TerraForma Database Schema
-- Run once on fresh database (auto-run via docker-entrypoint-initdb.d)

CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- Schema: registries — city business registry records
-- ============================================================
CREATE SCHEMA IF NOT EXISTS registries;

CREATE TABLE registries.businesses (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR(50) NOT NULL,
    name            TEXT NOT NULL,
    name_normalized TEXT,
    address         TEXT,
    address_normalized TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    category        TEXT,
    is_open         BOOLEAN NOT NULL,
    raw_source      JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_reg_city ON registries.businesses (city);
CREATE INDEX idx_reg_open ON registries.businesses (is_open);
CREATE INDEX idx_reg_name_norm ON registries.businesses (name_normalized);
CREATE INDEX idx_reg_geom ON registries.businesses
    USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326))
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- ============================================================
-- Schema: overture — Overture Maps POI records
-- ============================================================
CREATE SCHEMA IF NOT EXISTS overture;

CREATE TABLE overture.places (
    id              TEXT PRIMARY KEY,
    city            VARCHAR(50) NOT NULL,
    name            TEXT,
    name_normalized TEXT,
    category        TEXT,
    confidence      REAL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    address         TEXT,
    phone           TEXT,
    website         TEXT,
    raw_json        JSONB,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ov_city ON overture.places (city);
CREATE INDEX idx_ov_name_norm ON overture.places (name_normalized);
CREATE INDEX idx_ov_geom ON overture.places
    USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326));

CREATE TABLE overture.matched (
    id              SERIAL PRIMARY KEY,
    registry_id     INT REFERENCES registries.businesses(id),
    overture_id     TEXT REFERENCES overture.places(id),
    match_score     REAL,
    match_method    TEXT,
    is_open         BOOLEAN,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_match_reg ON overture.matched (registry_id);
CREATE INDEX idx_match_ov ON overture.matched (overture_id);

-- ============================================================
-- Schema: cv_scores — street imagery analysis results
-- ============================================================
CREATE SCHEMA IF NOT EXISTS cv_scores;

CREATE TABLE cv_scores.results (
    id                  SERIAL PRIMARY KEY,
    overture_id         TEXT REFERENCES overture.places(id),
    city                VARCHAR(50) NOT NULL,
    image_source        VARCHAR(20) NOT NULL,
    image_url           TEXT,
    image_captured_at   TIMESTAMP,
    cv_presence_score   REAL,
    raw_output          JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cv_overture ON cv_scores.results (overture_id);

-- ============================================================
-- Schema: web_scores — web crawling activity scores
-- ============================================================
CREATE SCHEMA IF NOT EXISTS web_scores;

CREATE TABLE web_scores.results (
    id                  SERIAL PRIMARY KEY,
    overture_id         TEXT REFERENCES overture.places(id),
    city                VARCHAR(50) NOT NULL,
    yelp_score          REAL,
    foursquare_score    REAL,
    tripadvisor_score   REAL,
    web_presence_score  REAL,
    composite_web_score REAL,
    raw_data            JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_web_overture ON web_scores.results (overture_id);

-- ============================================================
-- Schema: ground_truth — Google Places verification labels
-- ============================================================
CREATE SCHEMA IF NOT EXISTS ground_truth;

CREATE TABLE ground_truth.labels (
    id              SERIAL PRIMARY KEY,
    overture_id     TEXT REFERENCES overture.places(id),
    city            VARCHAR(50) NOT NULL,
    google_place_id TEXT,
    business_status TEXT,
    is_open         BOOLEAN,
    retrieved_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_gt_overture ON ground_truth.labels (overture_id);

-- ============================================================
-- Schema: predictions — final ensemble outputs
-- ============================================================
CREATE SCHEMA IF NOT EXISTS predictions;

CREATE TABLE predictions.results (
    id                      SERIAL PRIMARY KEY,
    overture_id             TEXT REFERENCES overture.places(id),
    city                    VARCHAR(50) NOT NULL,
    registry_pred_score     REAL,
    cv_presence_score       REAL,
    composite_web_score     REAL,
    ensemble_score          REAL,
    predicted_open          BOOLEAN,
    actual_open             BOOLEAN,
    correct                 BOOLEAN,
    created_at              TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_pred_overture ON predictions.results (overture_id);
CREATE INDEX idx_pred_city ON predictions.results (city);
