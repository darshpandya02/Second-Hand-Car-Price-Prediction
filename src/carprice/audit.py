"""Audit of the original 2022 notebook (notebooks/original/).

Reads the stored outputs of the notebook, recomputes the R² implied by its printed MSE
(the test split is deterministic: random_state=0, test_size=0.3), and re-runs its
modelling steps on the same 301-row file with the two fixes needed on current libraries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from . import data as D
from .metrics import all_metrics

NOTEBOOK = D.ROOT / "notebooks" / "original" / "Second_hand_car_price_prediction_system.ipynb"


def notebook_outputs(path: Path = NOTEBOOK) -> dict:
    nb = json.loads(path.read_text())
    text = []
    sources = []
    for cell in nb["cells"]:
        sources.append("".join(cell["source"]))
        for o in cell.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain") or ""
            text.append("".join(t))
    blob = "\n".join(text)
    src = "\n".join(sources)

    def grab(label):
        m = re.search(label + r"\s+([\d.]+)", blob)
        return float(m.group(1)) if m else None

    rows = re.search(r"RangeIndex: (\d+) entries", blob)
    cols = re.search(r"Data columns \(total (\d+) columns\)", blob)
    return {
        "csv_loaded": re.search(r"read_csv\('([^']+)'\)", src).group(1),
        "rows": int(rows.group(1)) if rows else None,
        "columns_after_dropping_name": int(cols.group(1)) if cols else None,
        "printed_mae": grab("Mean Absolute Error"),
        "printed_mse": grab("Mean Squared Error"),
        "printed_rmse": grab("Root Mean Absolute Error"),
        "mentions_r2": ("r2_score" in src) or ("R2" in blob) or (".score(" in src),
        "mentions_accuracy": "accuracy" in (src + blob).lower(),
        "mentions_tableau": "tableau" in (src + blob).lower(),
        "mentions_carwale": "carwale" in (src + blob).lower(),
        "execution_counts": sorted(
            {c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"},
            key=lambda v: (v is not None, v),
        ),
    }


def _legacy_frame() -> tuple[pd.DataFrame, pd.Series]:
    d = D.load_legacy().drop(columns=["Car_Name"])  # notebook: drop(columns=..., axis=1)
    d["years_old"] = 2022 - d["Year"]
    d = d.drop(columns=["Year"])
    d = pd.get_dummies(d, drop_first=True)
    return d.iloc[:, 1:], d.iloc[:, 0]


def rerun(seeds=(0, 1, 2, 3, 4)) -> dict:
    x, y = _legacy_frame()
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
    grid = {
        "n_estimators": [int(v) for v in np.linspace(100, 1200, 12)],
        "max_features": [1.0, "sqrt"],  # 'auto' was removed from sklearn in 1.3; 1.0 is equivalent
        "max_depth": [int(v) for v in np.linspace(5, 30, 6)],
        "min_samples_split": [2, 5, 10, 15, 100],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    runs = []
    for s in seeds:
        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=s, n_jobs=4),
            grid,
            scoring="neg_mean_squared_error",
            n_iter=10,
            cv=5,
            random_state=42,
            n_jobs=1,
        ).fit(x_tr, y_tr)
        m = all_metrics(y_te, search.predict(x_te))
        m["best_params"] = search.best_params_
        runs.append(m)
    r2s = np.array([r["r2"] for r in runs])
    return {
        "test_rows": int(len(y_te)),
        "test_variance": float(y_te.var(ddof=0)),
        "runs": runs,
        "r2_mean": float(r2s.mean()),
        "r2_min": float(r2s.min()),
        "r2_max": float(r2s.max()),
    }


def run(out_dir: Path) -> dict:
    nb = notebook_outputs()
    x, y = _legacy_frame()
    _, _, _, y_te = train_test_split(x, y, test_size=0.3, random_state=0)
    var = float(y_te.var(ddof=0))
    report = {
        "notebook": nb,
        "implied_test_r2_from_printed_mse": 1 - nb["printed_mse"] / var,
        "rerun_with_fixes": rerun(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    return report
