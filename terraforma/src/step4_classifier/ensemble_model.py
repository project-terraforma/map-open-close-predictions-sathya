"""
CatBoost + LightGBM ensemble with sklearn-compatible interface.

Adopted from StatusNow's approach: CatBoost handles native categoricals,
LightGBM provides diversity, weighted average of probabilities.

Improvements:
- Early stopping on both models to prevent overfitting
- Platt scaling (sigmoid calibration) for well-calibrated probabilities
- Optimal threshold search via balanced accuracy on validation data
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
        self._calibrator = None  # Platt scaling sigmoid

        self.cat_model = CatBoostClassifier(
            iterations=1200,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=7,
            random_seed=42,
            verbose=0,
            auto_class_weights="Balanced",
            early_stopping_rounds=50,
        )

        self.lgb_model = LGBMClassifier(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def fit(self, X, y, sample_weight=None, eval_set=None, **kwargs):
        """Train both models with early stopping if eval_set provided."""
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            from catboost import Pool
            train_pool = Pool(X, y)
            val_pool = Pool(X_val, y_val)
            self.cat_model.fit(train_pool, eval_set=val_pool, verbose=0)
            self.lgb_model.fit(
                X, y, sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[_lgb_early_stopping(50)],
            )
        else:
            self.cat_model.fit(X, y, verbose=0)
            self.lgb_model.fit(X, y, sample_weight=sample_weight)
        return self

    def calibrate(self, X_val, y_val):
        """Platt scaling: fit sigmoid on validation predictions for calibrated probabilities."""
        from sklearn.linear_model import LogisticRegression
        raw_proba = self.predict_proba(X_val)[:, 1].reshape(-1, 1)
        self._calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
        self._calibrator.fit(raw_proba, y_val)

        # Find optimal threshold on validation data
        from sklearn.metrics import balanced_accuracy_score
        cal_proba = self._calibrator.predict_proba(raw_proba)[:, 1]
        best_thresh, best_bal = 0.5, 0
        for t in np.arange(0.30, 0.70, 0.01):
            bal = balanced_accuracy_score(y_val, (cal_proba >= t).astype(int))
            if bal > best_bal:
                best_bal = bal
                best_thresh = t
        self.threshold = round(float(best_thresh), 2)
        return self

    def predict_proba(self, X):
        """Weighted average of both models' probabilities, with optional calibration."""
        p_cat = self.cat_model.predict_proba(X)
        p_lgb = self.lgb_model.predict_proba(X)
        raw = self.weight_cat * p_cat + self.weight_lgb * p_lgb
        if self._calibrator is not None:
            cal = self._calibrator.predict_proba(raw[:, 1].reshape(-1, 1))
            return cal
        return raw

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


def _lgb_early_stopping(stopping_rounds):
    """Create LightGBM early stopping callback."""
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
    except ImportError:
        # Older LightGBM versions
        return None
