// Browser (and Node) evaluator for the exported gradient-boosting bundle (format carprice-hgb-v1).
// Mirrors carprice.export.predict_bundle: categorical values are mapped to their integer codes,
// then every tree is walked; NaN follows missing_go_to_left, otherwise x <= threshold goes left.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CarPrice = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function encode(row, maps, features, categorical) {
    const x = new Float64Array(features.length);
    for (let j = 0; j < features.length; j++) {
      const name = features[j];
      const v = row[name];
      if (categorical.indexOf(name) >= 0) {
        const code = maps[name][String(v)];
        x[j] = code === undefined ? NaN : code;
      } else {
        x[j] = v === null || v === undefined || v === "" ? NaN : Number(v);
      }
    }
    return x;
  }

  function raw(model, x) {
    let total = model.baseline;
    const trees = model.trees;
    for (let k = 0; k < trees.length; k++) {
      const t = trees[k];
      let i = 0;
      while (!t.leaf[i]) {
        const xv = x[t.f[i]];
        if (Number.isNaN(xv)) i = t.m[i] ? t.l[i] : t.r[i];
        else if (xv <= t.t[i]) i = t.l[i];
        else i = t.r[i];
      }
      total += t.v[i];
    }
    return total;
  }

  function predict(bundle, row) {
    const f = bundle.features;
    const c = bundle.categorical;
    const price = Math.exp(raw(bundle.point, encode(row, bundle.maps.point, f, c)));
    const lo = Math.exp(raw(bundle.lo, encode(row, bundle.maps.lo, f, c)) - bundle.qhat);
    const hi = Math.exp(raw(bundle.hi, encode(row, bundle.maps.hi, f, c)) + bundle.qhat);
    return { price: price, lo: Math.min(lo, price), hi: Math.max(hi, price) };
  }

  return { predict: predict, encode: encode, raw: raw };
});
