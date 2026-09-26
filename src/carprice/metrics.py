"""Regression metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

METRIC_NAMES = ("r2", "mae", "mape", "rmse")


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.abs(y)))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


FUNCS = {"r2": r2, "mae": mae, "mape": mape, "rmse": rmse}


def all_metrics(y, p) -> dict:
    return {k: FUNCS[k](y, p) for k in METRIC_NAMES}


def bootstrap_ci(y, p, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap over test rows. Returns {metric: {"value", "lo", "hi"}}."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = {k: np.empty(n_boot) for k in METRIC_NAMES}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for k in METRIC_NAMES:
            samples[k][b] = FUNCS[k](y[idx], p[idx])
    a = (1 - level) / 2
    point = all_metrics(y, p)
    return {
        k: {
            "value": point[k],
            "lo": float(np.quantile(samples[k], a)),
            "hi": float(np.quantile(samples[k], 1 - a)),
        }
        for k in METRIC_NAMES
    }
