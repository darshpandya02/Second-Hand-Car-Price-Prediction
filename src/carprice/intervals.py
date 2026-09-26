"""Conformalized quantile regression (Romano et al., 2019) on the log-price scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import gradient_boosting


@dataclass
class CQR:
    alpha: float = 0.10  # target miscoverage, i.e. a 90% interval
    seed: int = 0

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        lo_q, hi_q = self.alpha / 2, 1 - self.alpha / 2
        self.lo_ = gradient_boosting(self.seed, loss="quantile", quantile=lo_q, max_iter=300)
        self.hi_ = gradient_boosting(self.seed, loss="quantile", quantile=hi_q, max_iter=300)
        self.lo_.fit(X_fit, y_fit)
        self.hi_.fit(X_fit, y_fit)
        log_y = np.log(np.asarray(y_cal, float))
        lo, hi = np.log(self.lo_.predict(X_cal)), np.log(self.hi_.predict(X_cal))
        scores = np.maximum(lo - log_y, log_y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat_ = float(np.quantile(scores, level, method="higher"))
        self.n_cal_ = n
        return self

    def predict_interval(self, X, point=None):
        lo = np.exp(np.log(self.lo_.predict(X)) - self.qhat_)
        hi = np.exp(np.log(self.hi_.predict(X)) + self.qhat_)
        if point is not None:
            lo, hi = np.minimum(lo, point), np.maximum(hi, point)
        return lo, hi
