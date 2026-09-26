// Parity test: the browser evaluator must reproduce the sklearn predictions stored at export time.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const { predict } = require("../../web/predict.js");

const dataDir = path.join(__dirname, "..", "..", "web", "data");
const bundle = JSON.parse(fs.readFileSync(path.join(dataDir, "model.json"), "utf8"));
const cases = JSON.parse(fs.readFileSync(path.join(dataDir, "parity_cases.json"), "utf8"));

test("JS predictions match sklearn", () => {
  let maxRel = 0;
  for (const c of cases) {
    const got = predict(bundle, c.input);
    for (const k of ["price", "lo", "hi"]) {
      maxRel = Math.max(maxRel, Math.abs(got[k] - c.expected[k]) / c.expected[k]);
    }
  }
  console.log(`cases=${cases.length} max relative difference=${maxRel}`);
  assert.ok(maxRel < 1e-9, `max relative difference ${maxRel}`);
});

test("missing specs and unknown model still give a finite ordered interval", () => {
  const row = { ...cases[0].input, model: "NotARealModel", engine: null, max_power: "", mileage: undefined };
  const out = predict(bundle, row);
  assert.ok(Number.isFinite(out.price));
  assert.ok(out.lo <= out.price && out.price <= out.hi);
});
