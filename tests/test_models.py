import numpy as np
import pandas as pd

from carprice import data as D
from carprice.export import export_bundle, predict_bundle, rows_from_frame
from carprice.intervals import CQR
from carprice.metrics import all_metrics, bootstrap_ci
from carprice.models import OrderedOrdinalEncoder, gradient_boosting, make_models


def _sample(n=600, seed=0):
    df = D.load_v3().sample(n, random_state=seed).reset_index(drop=True)
    return df[D.FEATURES], df[D.TARGET].values


def test_encoder_orders_by_target_and_maps_unknown_to_nan():
    X = pd.DataFrame({"c": ["a", "a", "b", "b", "z"], "n": [1, 2, 3, 4, 5]})
    y = np.array([1.0, 1.0, 10.0, 10.0, 5.0])
    enc = OrderedOrdinalEncoder(categorical=("c",), numeric=("n",), smoothing=0.0).fit(X, y)
    assert enc.mappings_["c"] == {"a": 0.0, "z": 1.0, "b": 2.0}
    out = enc.transform(pd.DataFrame({"c": ["b", "unseen"], "n": [1, 2]}))
    assert out[0, 0] == 2.0 and np.isnan(out[1, 0])


def test_all_models_fit_and_beat_nothing():
    X, y = _sample()
    for name, model in make_models().items():
        pred = model.fit(X, y).predict(X)
        assert pred.shape == y.shape and np.all(np.isfinite(pred)), name


def test_metrics_known_values():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    m = all_metrics(y, p)
    assert np.isclose(m["mae"], 1 / 3)
    assert np.isclose(m["rmse"], np.sqrt(1 / 3))
    assert np.isclose(m["mape"], (1 / 3) / 3)
    assert np.isclose(m["r2"], 1 - 1 / 2)
    ci = bootstrap_ci(y, p, n_boot=50)
    assert ci["mae"]["lo"] <= ci["mae"]["value"] <= ci["mae"]["hi"]


def test_export_reproduces_sklearn_exactly():
    X, y = _sample(800, seed=1)
    Xa, Xb, ya, yb = X.iloc[:500], X.iloc[500:650], y[:500], y[500:650]
    point = gradient_boosting(0, max_iter=60).fit(Xa, ya)
    cqr = CQR(alpha=0.2).fit(Xa, ya, Xb, yb)
    bundle = export_bundle(point, cqr.lo_, cqr.hi_, cqr.qhat_)
    Xt = X.iloc[650:]
    p = point.predict(Xt)
    lo, hi = cqr.predict_interval(Xt, point=p)
    ref = [predict_bundle(bundle, r) for r in rows_from_frame(Xt)]
    np.testing.assert_allclose([r["price"] for r in ref], p, rtol=1e-12)
    np.testing.assert_allclose([r["lo"] for r in ref], lo, rtol=1e-12)
    np.testing.assert_allclose([r["hi"] for r in ref], hi, rtol=1e-12)
    assert all(r["lo"] <= r["price"] <= r["hi"] for r in ref)


def test_missing_specs_are_handled():
    X, y = _sample()
    model = gradient_boosting(0, max_iter=30).fit(X, y)
    cqr = CQR(alpha=0.2).fit(X.iloc[:400], y[:400], X.iloc[400:], y[400:])
    bundle = export_bundle(model, cqr.lo_, cqr.hi_, cqr.qhat_)
    row = rows_from_frame(X.iloc[:1])[0]
    row.update(mileage=None, engine=None, max_power=None, seats=None, model="NotARealModel")
    out = predict_bundle(bundle, row)
    assert np.isfinite(out["price"]) and out["lo"] <= out["price"] <= out["hi"]
