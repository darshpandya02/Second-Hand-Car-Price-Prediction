"""Model definitions. Every model predicts the selling price in rupees."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC

N_JOBS = 4


class OrderedOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Encode each categorical column as an integer rank of its smoothed mean target.

    Ranking categories by (log) price lets tree splits on the code act like splits on
    "cheaper vs more expensive make/model". It is fitted on training folds only.
    Unseen categories become NaN, which the tree models route like missing values.
    The mapping is a plain dict, so the browser can reproduce it exactly.
    """

    def __init__(self, categorical=tuple(CATEGORICAL), numeric=tuple(NUMERIC), smoothing=5.0):
        self.categorical = categorical
        self.numeric = numeric
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("OrderedOrdinalEncoder needs y")
        y = np.asarray(y, dtype=float)
        prior = float(y.mean())
        self.mappings_ = {}
        for col in self.categorical:
            stats = pd.DataFrame({"c": X[col].astype(str).values, "y": y}).groupby("c")["y"]
            n, s = stats.count(), stats.sum()
            smoothed = (s + self.smoothing * prior) / (n + self.smoothing)
            order = smoothed.sort_values(kind="mergesort").index
            self.mappings_[col] = {cat: float(i) for i, cat in enumerate(order)}
        self.feature_names_ = list(self.categorical) + list(self.numeric)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.categorical:
            m = self.mappings_[col]
            cols.append(X[col].astype(str).map(m).astype(float).values)
        for col in self.numeric:
            cols.append(pd.to_numeric(X[col], errors="coerce").astype(float).values)
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


def _log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def median_baseline() -> DummyRegressor:
    """Predicts the training-set median price for every car."""
    return DummyRegressor(strategy="median")


def linear_model(alpha: float = 1.0) -> TransformedTargetRegressor:
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_log_km, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), CATEGORICAL),
            ("num", numeric, NUMERIC),
        ]
    )
    return _log_target(Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))]))


def _log_km(X):
    X = np.array(X, dtype=float, copy=True)
    km_idx = NUMERIC.index("km_driven")
    X[:, km_idx] = np.log1p(X[:, km_idx])
    return X


def random_forest(seed: int = 0, **params) -> TransformedTargetRegressor:
    rf_params = dict(n_estimators=400, max_features=0.5, min_samples_leaf=1)
    rf_params.update(params)
    return _log_target(
        Pipeline(
            [
                ("enc", OrderedOrdinalEncoder()),
                ("impute", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(random_state=seed, n_jobs=N_JOBS, **rf_params)),
            ]
        )
    )


def gradient_boosting(seed: int = 0, loss: str = "squared_error", quantile=None, **params):
    hgb_params = dict(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
    )
    hgb_params.update(params)
    if quantile is not None:
        hgb_params["quantile"] = quantile
    return _log_target(
        Pipeline(
            [
                ("enc", OrderedOrdinalEncoder()),
                ("hgb", HistGradientBoostingRegressor(loss=loss, random_state=seed, **hgb_params)),
            ]
        )
    )


def make_models(seed: int = 0) -> dict:
    return {
        "Median baseline": median_baseline(),
        "Linear (ridge, log price)": linear_model(),
        "Random forest": random_forest(seed),
        "Gradient boosting (HGB)": gradient_boosting(seed),
    }


__all__ = [
    "FEATURES",
    "OrderedOrdinalEncoder",
    "gradient_boosting",
    "linear_model",
    "make_models",
    "median_baseline",
    "random_forest",
]
