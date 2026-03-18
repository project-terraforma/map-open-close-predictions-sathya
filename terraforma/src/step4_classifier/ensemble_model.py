"""
CatBoost + LightGBM ensemble with sklearn-compatible interface.

Adopted from StatusNow's approach: CatBoost handles native categoricals,
LightGBM provides diversity, weighted average of probabilities.
"""

import numpy as np
import pickle


class EnsembleClassifier:
    """CatBoost + LightGBM ensemble for open/closed classification."""

    def __init__(self, weight_cat=0.7, weight_lgb=0.3, threshold=0.5):
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier

        self.weight_cat = weight_cat
        self.weight_lgb = weight_lgb
        self.threshold = threshold

        self.cat_model = CatBoostClassifier(
            iterations=800,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=7,
            random_seed=42,
            verbose=0,
            auto_class_weights="Balanced",
        )

        self.lgb_model = LGBMClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Train both models. CatBoost ignores sample_weight (uses auto_class_weights)."""
        self.cat_model.fit(X, y, verbose=0)
        self.lgb_model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        """Weighted average of both models' probabilities."""
        p_cat = self.cat_model.predict_proba(X)
        p_lgb = self.lgb_model.predict_proba(X)
        return self.weight_cat * p_cat + self.weight_lgb * p_lgb

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    @property
    def feature_importances_(self):
        """Average feature importances from both models."""
        cat_imp = self.cat_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        # Normalize both to sum to 1
        cat_imp = cat_imp / (cat_imp.sum() + 1e-10)
        lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-10)
        return self.weight_cat * cat_imp + self.weight_lgb * lgb_imp

    @property
    def classes_(self):
        return self.cat_model.classes_
