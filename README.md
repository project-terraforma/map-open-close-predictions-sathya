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
- Trained on **6,367 labeled samples** (4,977 open, 1,390 closed)

## Training Data

The XGBoost model was trained on **6,367 Overture places** with known open/closed labels from 3 sources:

| Source | Samples | Closed Rate | Description |
|--------|---------|-------------|-------------|
| Overture Project C (original) | 3,179 | 8.7% | Overture's own labeled sample dataset — mostly open businesses |
| Overture Project C (updated) | 2,740 | 39.7% | Updated batch with more balanced closed representation |
| Yelp API | 448 | 5.8% | Yelp `is_closed` field for businesses matched to Overture |

Each sample has 8 raw Overture features (confidence, source_age_days, has_website, has_phone, has_brand, address_complete, category, fields_populated) which get expanded into 19 features via engineering. Only 448 samples (7%) have Yelp rating/review data — the rest use NaN (XGBoost handles missing values natively).

The **metamodel** was evaluated on 407 test samples across 5 cities using Leave-One-City-Out cross-validation, where all 6 signals (XGBoost, Foursquare, Website, Yelp, Text, TomTom) score each business independently.

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

## Recent Improvements

- **SMOTE oversampling**: Addresses the 3.6:1 class imbalance (4,977 open vs 1,390 closed) by synthetically generating minority-class samples, improving closed-business detection
- **Early stopping**: XGBoost now uses early stopping (30 rounds) during both grid search and final training to prevent overfitting
- **Expanded hyperparameter search**: Added deep-tree + strong regularization and shallow-wide ensemble configs to the grid search
- **SMOTE-aware cross-validation**: SMOTE is applied per-fold during CV (only on training splits) to avoid data leakage

## Project Structure

```
training/                  # Model training
  train_xgboost.py         # XGBoost classifier (19 features, grid search, Platt scaling, SMOTE)
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
pip install xgboost scikit-learn imbalanced-learn
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

## Limitations

### Training Data
- **Small dataset**: 6,367 samples is small for a model meant to generalize to 100M+ places. Most ML models for this task would use 50k-500k labeled samples
- **Class imbalance**: 3.6:1 open-to-closed ratio means the model sees far fewer closed businesses. SMOTE helps but synthetic samples aren't real-world closed businesses
- **No city labels**: All 6,367 training samples lack city metadata — the model can't learn geographic patterns (e.g., NYC restaurants close faster than rural ones)
- **Overture-labeled data quality**: The bulk of training data (5,919 samples) comes from Overture's own Project C labeled sets, which may have labeling inconsistencies or biases toward certain business types
- **Yelp bias**: The 448 Yelp-labeled samples skew heavily open (94.2% open) because Yelp's API mostly returns active businesses. Closed businesses get delisted, so the Yelp source underrepresents closures
- **Category skew**: Hotels (356 samples) and professional services (201) are overrepresented. Common categories like "restaurant" only have 96 samples — the model likely performs worse on underrepresented categories

### Model
- **XGBoost model-only is weak**: 51.8% baseline accuracy (barely better than a coin flip) means the model alone isn't useful — it relies on the 6-signal ensemble to reach 85%+
- **No temporal features**: The model sees a single snapshot of each business. It can't detect *changes* over time (e.g., a place that just lost its phone number vs one that never had one)
- **OCR signal dropped**: Mapillary street-level imagery was too outdated to be useful — this was meant to be a strong signal but contributed nothing
- **TomTom near-zero weight**: TomTom's signal (0.006 weight) adds almost no value. Effectively a 5-signal ensemble
- **Test set is small**: 407 test samples across 5 US cities. Performance on international cities, rural areas, or non-English businesses is unknown
- **Metamodel is simple**: Logistic regression can't learn non-linear signal interactions (e.g., "Foursquare says open BUT website is dead" should be weighted differently than either signal alone)

### Signals
- **API dependency at inference**: Unlike approach 2, this approach needs to call Foursquare, Yelp, and website-check APIs for every prediction. This is expensive at scale and adds latency
- **Foursquare deprecation risk**: Foursquare has changed API access terms before — the signal could break if they restrict access
- **Website liveness is noisy**: A dead website doesn't always mean a closed business (site might be temporarily down), and a live website doesn't always mean open (abandoned sites stay up for years)

## Future Improvements

- **More training data**: 6,367 samples is too small. Scraping 50k+ Yelp/Google labels across more cities and categories would directly improve generalization
- **Overture release deltas**: Diff consecutive Overture releases (free) — places that lose sources, change categories, or drop confidence between releases are strong closure signals. This is the single highest-ROI improvement
- **Google Places signal**: Google's `business_status` field would be the single highest-accuracy signal, but requires API costs at ~$5/1000 lookups
- **Temporal features**: Track how features change over time (e.g., a place losing its phone number between releases is more predictive than never having one)
- **Category-specific models**: Train separate models for food/retail/services — closure patterns differ significantly by industry (restaurants close at ~15%, hospitals at ~1%)
- **Active learning**: Instead of random signal lookups, prioritize checking businesses where the model is least confident to maximize label value per API call
- **Metamodel upgrade**: Replace logistic regression with a gradient-boosted metamodel that can learn non-linear signal interactions
- **Better Yelp sampling**: Actively search for closed businesses on Yelp (filter by "closed" status) to balance the Yelp label source
- **International training data**: Current data is US-only. Adding European/Asian labeled data would test whether the model transfers across geographies

## Test Cities

| City | Test Samples |
|------|-------------|
| San Francisco | 76 |
| Los Angeles | 76 |
| Chicago | 76 |
| Miami | 68 |
| Philadelphia | 111 |
| **Total** | **407** |
