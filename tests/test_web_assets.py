"""Checks on the committed browser assets (rebuild with `uv run carprice train`)."""

import json

import pytest

from carprice.data import ROOT
from carprice.export import predict_bundle

WEB = ROOT / "web" / "data"


@pytest.fixture(scope="module")
def bundle():
    return json.loads((WEB / "model.json").read_text())


def test_bundle_matches_committed_sklearn_predictions(bundle):
    cases = json.loads((WEB / "parity_cases.json").read_text())
    assert len(cases) >= 100
    for c in cases:
        got = predict_bundle(bundle, c["input"])
        for k in ("price", "lo", "hi"):
            assert got[k] == pytest.approx(c["expected"][k], rel=1e-9)


def test_options_cover_bundle_categories(bundle):
    opts = json.loads((WEB / "options.json").read_text())
    brands = set(opts["brands"])
    assert brands <= set(bundle["maps"]["point"]["brand"])
    assert set(opts["fuels"]) <= set(bundle["maps"]["point"]["fuel"])
    models = {m for b in opts["brands"].values() for m in b}
    assert models <= set(bundle["maps"]["point"]["model"])


def test_trends_have_all_views():
    trends = json.loads((WEB / "trends.json").read_text())
    assert set(trends["views"]) == {"All", "Manual", "Automatic"}
    assert trends["views"]["All"]["Diesel"]["by_year"]
