# TerraForma - Approach 2: CatBoost + LightGBM Pipeline

This branch contains **Approach 2** of TerraForma — a PostGIS-backed pipeline that uses a CatBoost + LightGBM ensemble to predict whether Overture Maps businesses are open or permanently closed.

> **Approach 1** (6-signal metamodel with XGBoost) is on the `main` branch.

## Key Differences from Approach 1

| | Approach 1 (main) | Approach 2 (this branch) |
|---|---|---|
| **Model** | XGBoost + logistic regression metamodel | CatBoost (70%) + LightGBM (30%) ensemble |
| **Features** | 19 features | 45+ features (staleness, deltas, identity changes, interactions) |
| **Signals at inference** | 6 signals run live (XGBoost, Foursquare, Website, Yelp, etc.) | Model-only — signals used only for training labels |
| **Training labels** | 6,367 Yelp-labeled samples | Overture pre-labeled + iterative signal labels |
| **Infrastructure** | Flat files (JSON) | PostGIS database, Docker, 9-step pipeline |
| **Feedback loop** | No | Yes — web crawl + LLM (Llama) verification |

## Results

| Metric | Baseline (Overture only) | After Yelp Retraining |
|--------|-------------------------|----------------------|
| Balanced Accuracy | 67.4% | 70.2% |
| Closed Recall | 71.8% | 80.6% |
| AUC | 0.748 | 0.761 |

## Recent Improvements

- **Early stopping** on both CatBoost and LightGBM (50 rounds patience) to prevent overfitting
- **Platt scaling calibration** for well-calibrated probability outputs
- **Optimal threshold search** via balanced accuracy on calibration holdout
- **Deeper trees** (depth 6, 1200 iterations) with lower learning rate (0.03) for better generalization
- **Calibration holdout** split: 80% train / 20% calibration before final full-data retrain

## Quick Start

```bash
# Start PostGIS
docker compose -f terraforma/docker-compose.yml up -d

# Install
pip install -e "terraforma/[dev]"

# Train baseline model
python -m src.step4_classifier

# Run iterative retraining
python -m src.step4_classifier retrain

# Evaluate
python -m src.step4_classifier eval all

# Score all matched DB records
python -m src.step4_classifier predict
```

## Pipeline Steps

```
Step 1: Ingest business registries (SF, NYC, Paris, Singapore)
Step 2: Download Overture places via DuckDB + S3
Step 3: Fuzzy-match registries <-> Overture places
Step 4: Train CatBoost+LightGBM classifier + iterative retraining
Step 5: Cross-validation
Step 6: Website liveness checks
Step 7: Build ground truth labels
Step 8: 6-signal ensemble scoring
Step 9: Evaluation
```

## 45+ Engineered Features

All derived from Overture's published schema — no external data needed at inference:

- **Confidence**: `base_conf`, `base_conf_sq`
- **Sources/Recency**: `num_sources`, `log_num_sources`, `is_cross_verified`, `source_has_msft`, `source_has_meta`, `log_days`, `is_stale_3mo/6mo/1yr/2yr`, `recency_bucket`, `recency_spread`
- **Digital presence**: `has_website`, `has_phone`, `has_social`, `contact_depth`, `has_facebook`, `has_instagram`, `has_yelp`, `total_digital`
- **Deltas** (between Overture releases): `delta_websites`, `delta_socials`, `delta_phones`, `has_any_loss`, `contact_loss_severity`
- **Identity changes**: `name_changed`, `cat_changed`, `address_changed`, `website_domain_changed`
- **Category risk**: `cat_food_drink`, `cat_retail`, `cat_services`, `cat_health`, `cat_entertainment`, `cat_accommodation`
- **Interactions**: `zombie_score`, `nonbrand_stale_risk`, `food_x_stale`, `stale_x_low_conf`, `multi_signal_risk`

## Future Improvements

- **More Overture release pairs**: Currently using one pair of releases for delta features. Using 3-4 consecutive releases would capture velocity of change (accelerating loss = stronger closure signal)
- **Temporal decay weighting**: Weight recent training samples higher than old ones — closure patterns shift over time
- **Geographic features**: Cluster businesses by location — if 5 businesses at the same address all lose sources, it's likely a building closure
- **LLM-augmented labels**: Use Llama/GPT to verify ambiguous web crawl results for higher-quality feedback labels
- **Ensemble stacking**: Add a Random Forest or neural net as a third model to the CatBoost+LightGBM ensemble for more diversity
- **Feature selection**: Use SHAP values to prune low-importance features and reduce overfitting risk
- **Cross-city transfer learning**: Pre-train on data-rich cities (SF, NYC), fine-tune on data-sparse cities
- **Overture release diffing**: Compute feature deltas across 3+ releases to detect velocity of decline (a place losing 2 sources over 3 months is worse than losing 1)

## Project Structure

See [terraforma/README.md](terraforma/README.md) for full details on the pipeline architecture.
