/* global CarPrice, Chart */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { bundle: null, options: null, trends: null, charts: {} };
  const FUEL_COLORS = { Diesel: "#1f5fbf", Petrol: "#d9480f", CNG: "#2b8a3e", LPG: "#7048e8" };

  function formatINR(v) {
    if (v >= 1e7) return "₹" + (v / 1e7).toFixed(2) + " crore";
    return "₹" + (v / 1e5).toFixed(2) + " lakh";
  }
  function fullINR(v) {
    return "₹" + Math.round(v).toLocaleString("en-IN");
  }

  function fillSelect(sel, items, selected) {
    sel.innerHTML = "";
    for (const it of items) {
      const o = document.createElement("option");
      if (typeof it === "object") { o.value = it.value; o.textContent = it.label; }
      else { o.value = it; o.textContent = it; }
      sel.appendChild(o);
    }
    if (selected !== undefined && items.some((it) => String(typeof it === "object" ? it.value : it) === String(selected))) {
      sel.value = String(selected);
    }
  }

  function currentModelInfo() {
    const b = state.options.brands[$("brand").value] || {};
    return b[$("model").value];
  }

  function onBrandChange() {
    const models = state.options.brands[$("brand").value];
    const names = Object.keys(models).sort((a, b) => models[b].n - models[a].n || a.localeCompare(b));
    fillSelect($("model"), names.map((n) => ({ value: n, label: n + " (" + models[n].n + " listings)" })), names[0]);
    onModelChange();
  }

  function onModelChange() {
    const info = currentModelInfo();
    if (!info) return;
    const fuel = info.fuels.indexOf($("fuel").value) >= 0 ? $("fuel").value : info.fuels[0];
    fillSelect($("fuel"), state.options.fuels, fuel);
    const tr = info.transmissions.indexOf($("transmission").value) >= 0 ? $("transmission").value : info.transmissions[info.transmissions.length - 1];
    fillSelect($("transmission"), state.options.transmissions, tr);
    const y = Math.round((info.year_min + info.year_max) / 2);
    const cur = Number($("year").value);
    if (!(cur >= info.year_min && cur <= info.year_max)) $("year").value = Math.max(info.year_min, Math.min(info.year_max, y + 2));
    fillSpecs();
  }

  function fillSpecs() {
    const info = currentModelInfo();
    if (!info) return;
    const s = (info.specs_by_fuel && info.specs_by_fuel[$("fuel").value]) || info.specs;
    for (const k of ["engine", "max_power", "mileage", "seats"]) $(k).value = s[k] === null ? "" : s[k];
  }

  function readForm() {
    const num = (id) => ($(id).value === "" ? null : Number($(id).value));
    return {
      brand: $("brand").value,
      model: $("model").value,
      fuel: $("fuel").value,
      seller_type: $("seller_type").value,
      transmission: $("transmission").value,
      year: num("year"),
      km_driven: num("km_driven"),
      owner: num("owner"),
      mileage: num("mileage"),
      engine: num("engine"),
      max_power: num("max_power"),
      seats: num("seats"),
    };
  }

  function showPrediction(ev) {
    if (ev) ev.preventDefault();
    const row = readForm();
    const out = CarPrice.predict(state.bundle, row);
    $("price").textContent = formatINR(out.price);
    $("price").dataset.value = String(out.price);
    $("interval").textContent = "90% interval: " + formatINR(out.lo) + " to " + formatINR(out.hi) +
      " (" + fullINR(out.lo) + " to " + fullINR(out.hi) + ")";
    const span = out.hi - out.lo || 1;
    const pad = span * 0.1;
    const lo = out.lo - pad, hi = out.hi + pad;
    const pct = (v) => ((v - lo) / (hi - lo)) * 100;
    $("bar-fill").style.left = pct(out.lo) + "%";
    $("bar-fill").style.width = pct(out.hi) - pct(out.lo) + "%";
    $("bar-point").style.left = pct(out.price) + "%";

    const notes = [];
    const info = currentModelInfo();
    if (info && info.n < 10) notes.push("Only " + info.n + " listings of this model in the data, so treat the estimate with extra caution.");
    if (info && (row.year < info.year_min || row.year > info.year_max)) {
      notes.push("The data only has this model from " + info.year_min + " to " + info.year_max + ".");
    }
    if (info && info.fuels.indexOf(row.fuel) < 0) notes.push("No " + row.fuel + " listings of this model in the data.");
    if (row.km_driven > state.options.km_max) notes.push("Very high kilometres; few comparable listings.");
    $("notes").innerHTML = "";
    for (const n of notes) { const li = document.createElement("li"); li.textContent = n; $("notes").appendChild(li); }
    $("result").hidden = false;
    $("result").dataset.ready = "1";
  }

  // ---- market trends ------------------------------------------------------------------------

  function makeChart(canvas, xType, labels) {
    return new Chart(canvas, {
      type: "line",
      data: { labels: labels, datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: xType === "linear" ? { type: "linear", ticks: { callback: (v) => String(v) }, title: { display: true, text: "Model year" } }
            : { type: "category", title: { display: true, text: "Kilometres driven" } },
          y: { type: "logarithmic", title: { display: true, text: "Median price (₹ lakh)" },
            ticks: { callback: (v) => { const l = v / 1e5; return [0.5, 1, 2, 3, 5, 10, 20, 50].indexOf(+l.toFixed(1)) >= 0 ? l : ""; } } },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const r = ctx.raw;
                return ctx.dataset.label + ": " + formatINR(r.y) + " median, IQR " + formatINR(r.p25) + " to " + formatINR(r.p75) + ", n=" + r.n;
              },
            },
          },
        },
      },
    });
  }

  function updateTrends() {
    const view = state.trends.views[$("trend-trans").value];
    const logY = $("trend-log").checked;
    const fuels = Object.keys(view).sort((a, b) => view[b].n - view[a].n);
    const build = (key, isYear) => fuels.filter((f) => view[f][key].length).map((f) => ({
      label: f + " (n=" + view[f].n + ")",
      borderColor: FUEL_COLORS[f] || "#555",
      backgroundColor: FUEL_COLORS[f] || "#555",
      pointRadius: 3,
      tension: 0.2,
      spanGaps: true,
      parsing: false,
      data: view[f][key].map((p) => ({ x: isYear ? p.x : state.trends.km_labels.indexOf(p.x), y: p.median, p25: p.p25, p75: p.p75, n: p.n })),
    }));
    for (const [id, key, isYear] of [["year", "by_year", true], ["km", "by_km", false]]) {
      const ch = state.charts[id];
      ch.data.datasets = build(key, isYear);
      ch.options.scales.y.type = logY ? "logarithmic" : "linear";
      ch.options.scales.y.ticks.callback = logY
        ? (v) => { const l = v / 1e5; return [0.5, 1, 2, 3, 5, 10, 20, 50].indexOf(+l.toFixed(1)) >= 0 ? l : ""; }
        : (v) => (v / 1e5).toFixed(0);
      ch.update();
    }
    document.body.dataset.chartsReady = "1";
  }

  function renderMetrics(summary) {
    const rows = summary.models.map((m) =>
      "<tr><td>" + m.name + "</td><td>" + m.r2.toFixed(3) + "</td><td>" + (m.mae / 1e5).toFixed(2) +
      "</td><td>" + (m.mape * 100).toFixed(1) + "%</td><td>" + (m.rmse / 1e5).toFixed(2) + "</td></tr>").join("");
    $("metrics").innerHTML =
      "<p class='muted'>Held-out test set of " + summary.test_rows.toLocaleString("en-IN") + " listings. The page uses the gradient-boosting model. " +
      "MAE and RMSE in ₹ lakh.</p><table><thead><tr><th>Model</th><th>R²</th><th>MAE</th><th>MAPE</th><th>RMSE</th></tr></thead><tbody>" +
      rows + "</tbody></table>";
    $("cov").textContent = (summary.coverage * 100).toFixed(1) + "%";
    $("n-rows").textContent = summary.rows_used.toLocaleString("en-US");
  }

  async function init() {
    const get = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });
    const [options, trends, summary] = await Promise.all([get("data/options.json"), get("data/trends.json"), get("data/summary.json")]);
    state.options = options;
    state.trends = trends;
    renderMetrics(summary);

    const brands = Object.keys(options.brands).sort();
    fillSelect($("brand"), brands, "Maruti");
    fillSelect($("owner"), options.owners, 1);
    fillSelect($("seller_type"), options.seller_types, "Individual");
    $("brand").addEventListener("change", onBrandChange);
    $("model").addEventListener("change", onModelChange);
    $("fuel").addEventListener("change", fillSpecs);
    onBrandChange();

    state.charts.year = makeChart($("chart-year"), "linear");
    state.charts.km = makeChart($("chart-km"), "category", trends.km_labels);
    $("trend-trans").addEventListener("change", updateTrends);
    $("trend-log").addEventListener("change", updateTrends);
    updateTrends();

    state.bundle = await get("data/model.json");
    $("predict-btn").disabled = false;
    $("predict-btn").textContent = "Estimate price";
    $("car-form").addEventListener("submit", showPrediction);
    document.body.dataset.modelReady = "1";
  }

  init().catch((err) => {
    $("predict-btn").textContent = "Failed to load: " + err.message;
    console.error(err);
  });
})();
