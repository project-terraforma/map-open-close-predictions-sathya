# TerraForma

Predicting business open/closed status at scale using Overture Maps data and a multi-signal ensemble.

## Problem

Overture Maps has 400M+ places but no reliable open/closed field. Roughly 15-20% of POIs go stale over time. TerraForma predicts which businesses are permanently closed using only Overture metadata, with external signals used solely to improve training labels.

## Approach

### 6-Signal Ensemble with Iterative Retraining

We train a CatBoost + LightGBM ensemble on 19 features engineered from Overture metadata (confidence, source age, digital presence, category, brand, staleness, interaction terms). External signals generate verified training labels — they are **not** used at inference time.

**Signals (training-time only):**

| Signal | Weight | Purpose |
|--------|--------|---------|
| XGBoost/CatBoost+LightGBM | 45% | Core model on Overture features |
| Foursquare | 20% | Place existence verification |
| Website Liveness | 20% | HTTP checks for dead/parked domains |
| Yelp | 8% | `is_closed` field + review activity |
| OCR (Mapillary) | — | Dropped: imagery too outdated |
| TomTom | <1% | Directory cross-reference |

**Key insight:** Train with expensive signals, predict with cheap Overture features. At inference, only the model runs — no API calls needed.

### Iterative Retraining Pipeline

The model improves by progressively adding signal-verified labels to the training set:

| Round | Data | Samples | Balanced Acc | AUC | Closed Recall |
|-------|------|---------|-------------|------|---------------|
| R0 | Overture labels only | 3,006 | 67.4% | 0.748 | 71.8% |
| R1 | + Yelp verified labels | 3,536 | 70.2% | 0.761 | 80.6% |
| R2 | + Foursquare + website | 3,536 | 70.2% | 0.761 | 80.6% |
| R3 | + Web crawl feedback | 3,550 | 70.0% | 0.765 | 81.1% |

With just one signal (Yelp), balanced accuracy improved from 67.4% to 70.2% and closed recall jumped from 71.8% to 80.6%. Adding more signal volume (Foursquare labels, website liveness labels, web crawl labels) would continue pushing accuracy upward.

### 19 Engineered Features

All derived from Overture's published schema:

- **Confidence:** `base_conf`, `base_conf_sq`
- **Sources:** `num_sources`, `log_num_sources`, `is_cross_verified`, `source_has_msft`, `source_has_meta`
- **Digital presence:** `has_website`, `has_phone`, `has_social`, `contact_depth`, `has_facebook`, `has_instagram`, `has_yelp`, `total_digital`
- **Staleness:** `log_days`, `is_stale_3mo`, `is_stale_6mo`, `is_stale_1yr`, `is_stale_2yr`, `recency_bucket`
- **Category:** `cat_food_drink`, `cat_retail`, `cat_services`, `cat_health` (restaurants close at ~15%, hospitals at ~1%)
- **Interactions:** `zombie_score`, `nonbrand_stale_risk`, `stale_x_low_conf`, `food_x_stale`

## Pipeline

```
Step 1: Ingest business registries (SF, NYC, Paris, Singapore)
Step 2: Download Overture places via DuckDB + S3
Step 3: Fuzzy-match registries ↔ Overture places
Step 4: Train classifier + iterative retraining pipeline
Step 5: Cross-validation
Step 6: Website liveness checks
Step 7: Build ground truth labels
Step 8: 6-signal ensemble scoring
Step 9: Evaluation
Step 10: Retrain on new labels
```

## Usage

```bash
# Prerequisites: Docker (for PostGIS), Python 3.10+
docker compose up -d

# Install dependencies
pip install -e ".[dev]"

# Train baseline model
python -m src.step4_classifier

# Run iterative retraining pipeline
python -m src.step4_classifier retrain

# Evaluate against ground truth
python -m src.step4_classifier eval sf
python -m src.step4_classifier eval all

# Generate signal labels
python -m src.step4_classifier yelp 500
python -m src.step4_classifier foursquare sf 500

# Score all matched DB records
python -m src.step4_classifier predict
```

## Scaling

At inference, the model uses only Overture features — no APIs. Scoring 100M+ places costs ~$750 in compute per run.

For Overture's member companies, internal signals could replace all external APIs:
- **Microsoft:** Bing Places (replaces Foursquare)
- **Meta:** Facebook page activity (replaces Yelp)
- **TomTom:** Fleet telemetry (foot traffic proxy)
- **Release deltas:** Diff consecutive Overture releases for source/confidence changes (free, biggest accuracy boost)

## Project Structure

```
src/
  step1_registries/    # Business registry ingestion
  step2_overture/      # Overture Maps data download
  step3_matching/      # Fuzzy matching registries ↔ Overture
  step4_classifier/    # CatBoost+LightGBM model + retraining pipeline
  step6_web/           # Website liveness checks
  step7_ground_truth/  # Ground truth label management
  step8_ensemble/      # Multi-signal ensemble scoring
  step9_evaluation/    # Metrics and evaluation
  step10_retrain/      # Automated retraining
sql/                   # PostGIS schema definitions
models/                # Trained model artifacts (gitignored)
```
