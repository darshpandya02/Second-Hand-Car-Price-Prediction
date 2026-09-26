"""Export fitted HistGradientBoosting pipelines to a compact JSON format for the browser.

The JSON holds the categorical code maps and, for every tree, flat node arrays. The
browser evaluator (web/predict.js) walks these trees exactly as sklearn does:
NaN goes to the side given by ``missing_go_to_left``, otherwise ``x <= threshold`` goes left.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import FEATURES


def _hgb_parts(ttr):
    pipe = ttr.regressor_
    enc, hgb = pipe.named_steps["enc"], pipe.named_steps["hgb"]
    if getattr(hgb, "is_categorical_", None) is not None and np.any(hgb.is_categorical_):
        raise ValueError("native categorical splits are not supported by the exporter")
    return enc, hgb


def export_trees(ttr) -> dict:
    _, hgb = _hgb_parts(ttr)
    trees = []
    for predictors in hgb._predictors:
        (pred,) = predictors  # single output for regression
        nodes = pred.nodes
        trees.append(
            {
                "f": nodes["feature_idx"].astype(int).tolist(),
                "t": [float(v) for v in nodes["num_threshold"]],
                "l": nodes["left"].astype(int).tolist(),
                "r": nodes["right"].astype(int).tolist(),
                "m": nodes["missing_go_to_left"].astype(int).tolist(),
                "leaf": nodes["is_leaf"].astype(int).tolist(),
                "v": [float(v) for v in nodes["value"]],
            }
        )
    baseline = float(np.asarray(hgb._baseline_prediction).ravel()[0])
    return {"baseline": baseline, "trees": trees}


def export_bundle(point, lo, hi, qhat: float) -> dict:
    # The interval models are fitted on a sub-split, so each model keeps its own code map.
    enc, _ = _hgb_parts(point)
    return {
        "format": "carprice-hgb-v1",
        "target": "log(selling_price in INR)",
        "features": list(enc.feature_names_),
        "categorical": list(enc.categorical),
        "maps": {
            "point": enc.mappings_,
            "lo": _hgb_parts(lo)[0].mappings_,
            "hi": _hgb_parts(hi)[0].mappings_,
        },
        "point": export_trees(point),
        "lo": export_trees(lo),
        "hi": export_trees(hi),
        "qhat": qhat,
    }


# ---- reference evaluator (mirrors web/predict.js) -----------------------------------------


def _encode(row: dict, maps: dict, features: list, categorical: list) -> list:
    x = []
    for name in features:
        v = row.get(name)
        if name in categorical:
            code = maps[name].get(str(v))
            x.append(math.nan if code is None else code)
        else:
            x.append(math.nan if v is None or v == "" else float(v))
    return x


def _raw(model: dict, x: list) -> float:
    total = model["baseline"]
    for t in model["trees"]:
        i = 0
        while not t["leaf"][i]:
            xv = x[t["f"][i]]
            if math.isnan(xv):
                i = t["l"][i] if t["m"][i] else t["r"][i]
            elif xv <= t["t"][i]:
                i = t["l"][i]
            else:
                i = t["r"][i]
        total += t["v"][i]
    return total


def predict_bundle(bundle: dict, row: dict) -> dict:
    f, c = bundle["features"], bundle["categorical"]
    point = math.exp(_raw(bundle["point"], _encode(row, bundle["maps"]["point"], f, c)))
    lo = math.exp(_raw(bundle["lo"], _encode(row, bundle["maps"]["lo"], f, c)) - bundle["qhat"])
    hi = math.exp(_raw(bundle["hi"], _encode(row, bundle["maps"]["hi"], f, c)) + bundle["qhat"])
    return {"price": point, "lo": min(lo, point), "hi": max(hi, point)}


def rows_from_frame(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df[FEATURES].to_dict(orient="records"):
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()})
    return out
