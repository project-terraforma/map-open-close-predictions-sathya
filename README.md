# TerraForma - Business Open/Closed Prediction

Predicting whether businesses listed in Overture Maps are currently open or permanently closed, using a 6-signal ensemble approach built on top of Overture Places data.

## Overview

This project builds a metamodel that combines 6 independent signals to predict business status:

| Signal | Weight | Description |
|--------|--------|-------------|
| **XGBoost** | 2.258 (highest) | 19-feature model trained on Overture place attributes |
| **Foursquare** | 1.017 | Cross-references Foursquare venue data |
| **Website** | 1.008 | Checks if business website is alive/dead |
| **Yelp** | 0.420 | Yelp review activity and status |
| **Text/OCR** | 0.270 | Text signals (OCR from Mapillary was explored but dropped — imagery too outdated) |
| **TomTom** | 0.006 | TomTom POI cross-reference |

A **logistic regression metamodel** combines these signal scores into a final open/closed prediction.

### Results

- **85-93% accuracy** across 5 test cities (SF, LA, Chicago, Miami, Philadelphia)
- XGBoost model-only accuracy: **51.8% baseline -> 62.5% after retraining** with signal labels
- Trained on **6,367 Yelp-labeled samples** (4,977 open, 1,390 closed)

### XGBoost Features (19 total)

**Base features (10):** category present, has phone, has website, has email, source count, has social media, address completeness, has brand, name length, has hours

**Engineered features (9):** old source flag, sparse record, category closure rate, multi-source agreement, contact richness, chain indicator, address quality, digital presence, data completeness

Top feature by importance: `category_closure_rate` (28.1%)

## Iterative Retraining

The retraining pipeline uses high-confidence signal outputs as training labels to progressively improve the XGBoost model-only accuracy:

| Round | Training Samples | Avg Accuracy | Best Improvement |
|-------|-----------------|--------------|------------------|
| R0: Baseline | 6,367 | 51.8% | -- |
| R1: +Yelp labels | 6,655 | **62.5%** | Miami 47->73.5%, Philly 60->89.2% |
| R2: +Foursquare | 6,861 | 56.2% | -- |
| R3: +Website | 7,146 | 54.1% | -- |
| R4: +Metamodel | 7,369 | 60.0% | SF 50->60.5%, Miami 47->76.5% |

Yelp labels provided the single biggest accuracy boost. The full ensemble still outperforms model-only predictions, but retraining narrows the gap.

## Project Structure

```
training/                  # Model training
  train_xgboost.py         # XGBoost classifier (19 features, grid search, Platt scaling)
  train_metamodel.py        # Logistic regression metamodel over 6 signals
  retrain_pipeline.py       # Iterative retraining using signal labels
  feature_engineering.py    # Feature extraction from Overture place data

signals/                   # External signal checkers
  check_website_liveness.py
  check_facebook.py
  check_tomtom.py
  enrich_yelp.py
  ocr_model.py             # OCR from Mapillary (dropped)
  run_vision.py

scoring/                   # Prediction & optimization
  predict.py
  generate_predictions.py
  optimize_*.py            # Threshold optimization variants

data/
  ingest/                  # Data download & extraction
  labeling/                # Ground truth collection
  candidates/              # Overture candidate JSONs per city
  training_data/           # Training datasets (yelp_training_data.json)

model/                     # Saved models & weights
  metamodel.json           # Metamodel weights & LOCO-CV results
  xgboost_model.json
  xgb_feature_importance.json

evaluation/                # Evaluation outputs
  retrain_results.json     # Per-round retraining accuracy
  confusion_matrix.png
  feature_importance.png

analysis/                  # Error analysis & evaluation scripts
pipeline/                  # Pipeline orchestration
frontend/                  # React + Vite map visualization
tests/                     # Test files
```

## Usage

### Train XGBoost model
```bash
python training/train_xgboost.py
```

### Train metamodel
```bash
python training/train_metamodel.py
```

### Run iterative retraining
```bash
python training/retrain_pipeline.py
```
Results saved to `evaluation/retrain_results.json`.

### Generate predictions
```bash
python scoring/predict.py
```

### Run frontend map
```bash
cd frontend && npm install && npm run dev
```

## How It Scales

This approach is designed to scale to Overture's 100M+ places:

1. **XGBoost model** runs on Overture attributes alone -- no external API calls needed
2. **Signal ensemble** adds accuracy where external data is available
3. **Retraining pipeline** allows the model to learn from signal outputs, gradually reducing dependence on expensive API calls
4. **Per-city evaluation** ensures the model generalizes across geographies

## Test Cities

| City | Test Samples |
|------|-------------|
| San Francisco | 76 |
| Los Angeles | 76 |
| Chicago | 76 |
| Miami | 68 |
| Philadelphia | 111 |
| **Total** | **407** |
