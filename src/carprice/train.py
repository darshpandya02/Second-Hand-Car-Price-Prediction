"""End-to-end experiment: split, tune, cross-validate, test with bootstrap CIs, intervals, export."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, KFold, cross_validate, train_test_split

from . import data as D
from .export import export_bundle, predict_bundle, rows_from_frame
from .intervals import CQR
from .metrics import all_metrics, bootstrap_ci
from .models import N_JOBS, gradient_boosting, linear_model, median_baseline, random_forest
from .webdata import SPEC_COLS, build_options, build_trends

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_GRID = {
    "regressor__rf__max_features": [0.33, 0.5, 0.8],
    "regressor__rf__min_samples_leaf": [1, 2],
}
HGB_GRID = {
    "regressor__hgb__learning_rate": [0.05, 0.1],
    "regressor__hgb__max_leaf_nodes": [15, 31, 63],
    "regressor__hgb__min_samples_leaf": [5, 10],
}


def _tune(model, grid, X, y, cv):
    search = GridSearchCV(model, grid, scoring="r2", cv=cv, n_jobs=1, refit=False)
    search.fit(X, y)
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def _cv(model, X, y, cv):
    res = cross_validate(
        model,
        X,
        y,
        cv=cv,
        n_jobs=1,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
            "rmse": "neg_root_mean_squared_error",
        },
    )
    out = {}
    for k in ("r2", "mae", "mape", "rmse"):
        v = res[f"test_{k}"] * (1 if k == "r2" else -1)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    return out


def run(out_dir: Path, web_dir: Path, n_boot: int = 2000, quick: bool = False) -> dict:
    t0 = time.time()
    raw = pd.read_csv(D.V3_PATH)
    df = D.load_v3()
    X, y = df[D.FEATURES], df[D.TARGET].values
    X_tr, X_te, y_tr, y_te, name_tr, name_te = train_test_split(
        X, y, df["name"], test_size=TEST_SIZE, random_state=SEED
    )
    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)

    if quick:
        rf_best, hgb_best = {}, {}
        tuning = {}
    else:
        rf_best, rf_cv = _tune(random_forest(SEED, n_estimators=200), RF_GRID, X_tr, y_tr, cv)
        hgb_best, hgb_cv = _tune(gradient_boosting(SEED), HGB_GRID, X_tr, y_tr, cv)
        tuning = {
            "random_forest": {"grid": RF_GRID, "best": rf_best, "cv_r2": rf_cv},
            "gradient_boosting": {"grid": HGB_GRID, "best": hgb_best, "cv_r2": hgb_cv},
        }

    models = {
        "Median baseline": median_baseline(),
        "Linear (ridge, log price)": linear_model(),
        "Random forest": random_forest(SEED, **rf_best),
        "Gradient boosting (HGB)": gradient_boosting(SEED, **hgb_best),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        cv_scores = _cv(model, X_tr, y_tr, cv)
        m = clone(model).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        fitted[name] = m
        se = (y_te - pred) ** 2
        results[name] = {
            "cv": cv_scores,
            "test": bootstrap_ci(y_te, pred, n_boot=n_boot, seed=SEED),
            "sse_share_top10": float(np.sort(se)[-10:].sum() / se.sum()),
        }

    # Harder check: hold out whole listing titles so no variant is seen in training.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    g_tr, g_te = next(gss.split(X, y, groups=df["name"]))
    grouped = {}
    for name in ("Random forest", "Gradient boosting (HGB)"):
        m = clone(models[name]).fit(X.iloc[g_tr], y[g_tr])
        grouped[name] = all_metrics(y[g_te], m.predict(X.iloc[g_te]))

    # Permutation importance on the test set (R² drop).
    importance = {}
    for name in ("Random forest", "Gradient boosting (HGB)"):
        pi = permutation_importance(
            fitted[name], X_te, y_te, scoring="r2", n_repeats=10, random_state=SEED, n_jobs=1
        )
        importance[name] = sorted(
            (
                {"feature": f, "mean": float(mu), "std": float(sd)}
                for f, mu, sd in zip(D.FEATURES, pi.importances_mean, pi.importances_std, strict=True)
            ),
            key=lambda r: -r["mean"],
        )

    # 90% prediction intervals via conformalized quantile regression.
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, random_state=SEED)
    cqr = CQR(alpha=0.10, seed=SEED).fit(X_fit, y_fit, X_cal, y_cal)
    point_model = fitted["Gradient boosting (HGB)"]
    point_te = point_model.predict(X_te)
    lo, hi = cqr.predict_interval(X_te, point=point_te)
    covered = (y_te >= lo) & (y_te <= hi)
    intervals = {
        "method": "conformalized quantile regression (HGB quantile 0.05/0.95, log scale)",
        "target_coverage": 0.90,
        "n_calibration": cqr.n_cal_,
        "qhat_log": cqr.qhat_,
        "test_coverage": float(covered.mean()),
        "median_relative_width": float(np.median((hi - lo) / point_te)),
        "median_width_inr": float(np.median(hi - lo)),
    }

    # Export for the browser and check it reproduces sklearn.
    bundle = export_bundle(point_model, cqr.lo_, cqr.hi_, cqr.qhat_)
    rows = rows_from_frame(X_te)
    ref = [predict_bundle(bundle, r) for r in rows]
    rel = np.array([abs(r["price"] - p) / p for r, p in zip(ref, point_te, strict=True)])
    rel_lo = np.array([abs(r["lo"] - p) / p for r, p in zip(ref, lo, strict=True)])
    rel_hi = np.array([abs(r["hi"] - p) / p for r, p in zip(ref, hi, strict=True)])
    parity = {
        "n": len(rows),
        "max_rel_diff_point": float(rel.max()),
        "max_rel_diff_lo": float(rel_lo.max()),
        "max_rel_diff_hi": float(rel_hi.max()),
    }
    cases = [
        {"input": r, "name": n, "expected": {"price": float(p), "lo": float(a), "hi": float(b)}}
        for r, n, p, a, b in list(zip(rows, name_te, point_te, lo, hi, strict=True))[:300]
    ]

    report = {
        "dataset": {
            "source": "Kaggle nehalbirla/vehicle-dataset-from-cardekho, file 'Car details v3.csv'",
            "license": "Open Database License (ODbL) v1.0 for the database, "
            "Database Contents License (DbCL) v1.0 for the contents",
            "sha256": D.CHECKSUMS["car_details_v3.csv"],
            "raw_rows": int(len(raw)),
            "raw_columns": int(raw.shape[1]),
            "exact_duplicates_removed": int(len(raw) - len(df)),
            "rows_used": int(len(df)),
            "features": D.FEATURES,
            "n_features": len(D.FEATURES),
            "year_range": [int(df.year.min()), int(df.year.max())],
            "n_brands": int(df.brand.nunique()),
            "n_models": int(df.groupby(["brand", "model"]).ngroups),
            "missing_spec_rows": int(df[SPEC_COLS].isna().any(axis=1).sum()),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
        },
        "protocol": {
            "split": f"random {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, seed {SEED}",
            "cv": f"{CV_FOLDS}-fold KFold on the training split",
            "bootstrap": f"{n_boot} percentile resamples of the test set, 95% CI",
            "n_jobs": N_JOBS,
        },
        "tuning": tuning,
        "results": results,
        "grouped_by_title": {"test_rows": int(len(g_te)), "metrics": grouped},
        "permutation_importance": importance,
        "intervals": intervals,
        "export_parity": parity,
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (out_dir / "results.md").write_text(render_markdown(report))

    data_dir = web_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "model.json").write_text(json.dumps(bundle, separators=(",", ":")))
    # Dropdowns only offer brands and models the served model saw during training.
    train_frame = df.loc[X_tr.index]
    (data_dir / "options.json").write_text(json.dumps(build_options(train_frame), separators=(",", ":")))
    (data_dir / "trends.json").write_text(json.dumps(build_trends(df), separators=(",", ":")))
    (data_dir / "parity_cases.json").write_text(json.dumps(cases, separators=(",", ":")))
    summary = {
        "rows_used": report["dataset"]["rows_used"],
        "n_features": report["dataset"]["n_features"],
        "test_rows": report["dataset"]["test_rows"],
        "coverage": intervals["test_coverage"],
        "models": [
            {"name": name, **{k: r["test"][k]["value"] for k in ("r2", "mae", "mape", "rmse")}}
            for name, r in results.items()
        ],
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, separators=(",", ":")))
    return report


def _fmt(k, v):
    if k == "r2":
        return f"{v:.3f}"
    if k == "mape":
        return f"{v * 100:.1f}%"
    return f"{v / 1e5:.2f}"


def render_markdown(report: dict) -> str:
    lines = [
        "| Model | Test R² [95% CI] | MAE, ₹ lakh [95% CI] | MAPE [95% CI] "
        "| RMSE, ₹ lakh [95% CI] | CV R² (5-fold) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["results"].items():
        cells = []
        for k in ("r2", "mae", "mape", "rmse"):
            t = r["test"][k]
            cells.append(f"{_fmt(k, t['value'])} [{_fmt(k, t['lo'])}, {_fmt(k, t['hi'])}]")
        cv = r["cv"]["r2"]
        cells.append(f"{cv['mean']:.3f} ± {cv['std']:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    shares = ", ".join(f"{name}: {r['sse_share_top10'] * 100:.0f}%" for name, r in report["results"].items())
    lines.append(f"Share of test squared error from the 10 worst-predicted cars: {shares}.")
    lines.append("")
    lines.append("Grouped split (whole listing titles held out):")
    lines.append("")
    lines.append("| Model | R² | MAE, ₹ lakh | MAPE | RMSE, ₹ lakh |")
    lines.append("|---|---|---|---|---|")
    for name, m in report["grouped_by_title"]["metrics"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(k, m[k]) for k in ("r2", "mae", "mape", "rmse")) + " |")
    lines.append("")
    iv = report["intervals"]
    lines.append(
        f"90% conformal interval: test coverage {iv['test_coverage'] * 100:.1f}%, "
        f"median width {iv['median_relative_width'] * 100:.0f}% of the predicted price."
    )
    lines.append("")
    lines.append("Permutation importance (test R² drop, gradient boosting):")
    lines.append("")
    lines.append("| Feature | Mean drop | Std |")
    lines.append("|---|---|---|")
    for r in report["permutation_importance"]["Gradient boosting (HGB)"]:
        lines.append(f"| {r['feature']} | {r['mean']:.3f} | {r['std']:.3f} |")
    return "\n".join(lines) + "\n"
