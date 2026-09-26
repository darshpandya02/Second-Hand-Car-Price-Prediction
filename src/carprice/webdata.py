"""Static JSON for the web page: form options, per-model spec defaults, and market-trend aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import OWNER_MAP

SPEC_COLS = ["mileage", "engine", "max_power", "seats"]
KM_BINS = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 125_000, 150_000, 200_000, np.inf]
MIN_GROUP = 5


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    return v


def build_options(df: pd.DataFrame) -> dict:
    brands = {}
    for (brand, model), g in df.groupby(["brand", "model"], sort=True):
        specs_by_fuel = {}
        for fuel, gf in g.groupby("fuel"):
            specs_by_fuel[fuel] = {c: _clean(gf[c].median()) for c in SPEC_COLS}
        brands.setdefault(brand, {})[model] = {
            "n": int(len(g)),
            "year_min": int(g["year"].min()),
            "year_max": int(g["year"].max()),
            "fuels": sorted(g["fuel"].unique().tolist()),
            "transmissions": sorted(g["transmission"].unique().tolist()),
            "specs": {c: _clean(g[c].median()) for c in SPEC_COLS},
            "specs_by_fuel": specs_by_fuel,
        }
    return {
        "brands": brands,
        "fuels": sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
        "seller_types": sorted(df["seller_type"].unique().tolist()),
        "owners": [{"label": k, "value": v} for k, v in sorted(OWNER_MAP.items(), key=lambda kv: kv[1])],
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "km_max": int(df["km_driven"].quantile(0.999)),
        "n_rows": int(len(df)),
    }


def _series(g: pd.DataFrame, key: str, labels=None) -> list:
    out = []
    for k, gg in g.groupby(key, observed=True):
        if len(gg) < MIN_GROUP:
            continue
        p = gg["selling_price"]
        out.append(
            {
                "x": labels[k] if labels is not None else int(k),
                "median": float(p.median()),
                "p25": float(p.quantile(0.25)),
                "p75": float(p.quantile(0.75)),
                "n": int(len(gg)),
            }
        )
    return out


def build_trends(df: pd.DataFrame) -> dict:
    labels_list = []
    for lo, hi in zip(KM_BINS[:-1], KM_BINS[1:], strict=True):
        labels_list.append(f"{int(lo / 1000)}k+" if np.isinf(hi) else f"{int(lo / 1000)}-{int(hi / 1000)}k")
    d = df.copy()
    d["km_bin"] = pd.cut(d["km_driven"], KM_BINS, right=False, labels=False)
    labels = dict(enumerate(labels_list))
    out = {"km_labels": labels_list, "min_group": MIN_GROUP, "views": {}}
    for trans in ["All"] + sorted(d["transmission"].unique()):
        sub = d if trans == "All" else d[d["transmission"] == trans]
        view = {}
        for fuel, g in sub.groupby("fuel"):
            by_year = _series(g, "year")
            by_km = _series(g, "km_bin", labels)
            if by_year or by_km:
                view[fuel] = {"by_year": by_year, "by_km": by_km, "n": int(len(g))}
        out["views"][trans] = view
    return out
