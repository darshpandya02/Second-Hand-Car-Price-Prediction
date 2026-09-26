# Second-Hand Car Price Prediction

Predict the resale price of a used car in India from its make, model, year, kilometres, fuel,
transmission, ownership history and specifications.

**Live demo: https://used-car-price-india.vercel.app** (runs fully in the browser, no server).

This started as a single Jupyter notebook I wrote in 2022 (uploaded here in 2024). In 2026 I
audited it, found that it did not re-run and that several things I had said about it were not
supported by the code, and rebuilt it as a reproducible package with a proper evaluation,
calibrated prediction intervals, tests, CI and a web app.

## 1. Audit of the original notebook

The notebook is kept byte-for-byte under
[`notebooks/original/`](notebooks/original/Second_hand_car_price_prediction_system.ipynb).
`uv run carprice audit` reproduces the numbers below and writes `reports/audit.json`.

| Question | Finding |
|---|---|
| Data it loads | `car data.csv`, the 301-row file from the Kaggle "Vehicle dataset" by Nehal Birla (CarDekho listings). It is not scraped and it is not from CarWale. The CSV was never committed. |
| Size | 301 rows, 9 columns (8 after dropping `Car_Name`). Model years 2003 to 2018. |
| Model | Random forest, `RandomizedSearchCV` (10 candidates, 5-fold CV), 70/30 split with `random_state=0`, so 91 test cars. |
| Metrics printed | MAE 0.888 lakh, MSE 3.98, RMSE 2.00 lakh. No R², no "accuracy", no Tableau, no mention of CarWale. |
| R² actually achieved | Not printed. The split is deterministic, so the R² implied by the printed MSE is **0.867** (test variance 29.9). Re-running the same search with 5 seeds gives 0.864 to 0.869. |
| Does it re-run? | No, not as committed. The CSV is missing; on current pandas `drop(columns=..., axis=1)` raises, so `Car_Name` stays in and is one-hot encoded into 107 features; `DataFrame.corr()` fails on string columns; `max_features='auto'` was removed in scikit-learn 1.3, so every search candidate using it errors and is silently scored as NaN; the final `predict` cells then fail on a feature-count mismatch. With 2022-era libraries (pandas 1.4, scikit-learn 1.0) the modelling cells run and give MAE 0.899 and RMSE 2.03, close to the stored output (the forest was not seeded). |
| Leakage / design notes | `Present_Price` (ex-showroom price) is the dominant feature, which is fine for a valuation tool but makes the task easier than it looks. Test set of 91 cars means wide uncertainty. |

So the claims I had attached to this project (web-scraped CarWale data from 2018 to 2022,
100,000+ records, 15+ parameters, "92% accuracy", a Tableau dashboard) do not hold for the
original notebook. "Accuracy" is not a regression metric; the closest honest number was a test
R² of about 0.87 on 91 cars.

## 2. Data used in the rebuild

No scraping was done. The rebuild uses the larger file from the same public Kaggle dataset.

* Source: [Kaggle `nehalbirla/vehicle-dataset-from-cardekho`](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), version 4 (updated 2023-01-14), file `Car details v3.csv`.
* License: **Open Database License (ODbL) v1.0** for the database and **Database Contents License (DbCL) v1.0** for the contents, as listed on Kaggle. The two CSVs are redistributed in `data/raw/` under those terms with this attribution; derived data in `web/data/` is shared under the same terms. SHA-256 checksums are verified on load.
* Raw: 8,128 rows, 13 columns. 1,202 exact duplicate rows are removed so copies cannot sit in both train and test, leaving **6,926 listings**.
* **12 features**: brand and model (parsed from the title; 32 brands, 211 models), fuel, seller type, transmission, model year, km driven, owner count, mileage, engine cc, max power bhp, seats. `torque` is dropped because it mixes units in free text. 224 rows have missing specifications; the tree models handle them natively.
* Model years 1983 to 2020. Prices are asking prices in INR.

## 3. Method

* Random 80/20 train/test split (seed 42): 5,540 train, 1,386 test. All tuning and CV use the training split only.
* Target: log price for the linear and tree models; metrics are computed on rupees.
* Categorical columns are encoded by the rank of their smoothed mean log price (fitted inside each CV fold), which keeps the trees simple and exportable.
* Models: median baseline, ridge regression (one-hot, log km), random forest (the model named on the original project), and histogram gradient boosting. Random forest and gradient boosting are tuned with a small grid under 5-fold CV.
* Test metrics carry 95% percentile-bootstrap CIs (2,000 resamples).
* 90% prediction intervals: conformalized quantile regression (gradient-boosting quantile models at 5% and 95%, calibrated on a held-out 25% of the training split).
* Feature importance: permutation importance on the test set (drop in R²).
* Parallelism capped at 4 threads.

Reproduce everything with:

```bash
uv sync
uv run carprice audit    # original notebook audit, ~1 min
uv run carprice train    # full experiment + web assets, ~5 min
uv run pytest            # Python tests
node --test tests/js/predict.test.cjs   # browser evaluator parity
```

## 4. Results (test set, 1,386 listings)

| Model | Test R² [95% CI] | MAE, ₹ lakh [95% CI] | MAPE [95% CI] | RMSE, ₹ lakh [95% CI] | CV R² (5-fold) |
|---|---|---|---|---|---|
| Median baseline | -0.046 [-0.064, -0.032] | 2.60 [2.40, 2.81] | 79.0% [73.0%, 85.8%] | 4.79 [3.84, 5.79] | -0.052 ± 0.013 |
| Linear (ridge, log price) | 0.876 [0.810, 0.927] | 0.75 [0.68, 0.83] | 16.6% [15.7%, 17.5%] | 1.65 [1.16, 2.15] | 0.897 ± 0.009 |
| Random forest | 0.799 [0.704, 0.925] | 0.75 [0.66, 0.86] | 16.5% [15.6%, 17.5%] | 2.10 [1.07, 2.96] | 0.846 ± 0.070 |
| Gradient boosting (HGB) | 0.837 [0.768, 0.929] | 0.73 [0.65, 0.82] | 15.7% [14.9%, 16.7%] | 1.89 [1.05, 2.60] | 0.860 ± 0.076 |

The metric that replaces "accuracy" is **R² on the held-out test set**, reported together with
MAPE (the typical percentage error for a single car) because R² on rupees is dominated by a few
expensive cars.

What the table says:

* Gradient boosting has the lowest typical error (MAPE 15.7%, MAE ₹0.73 lakh) and is the model served on the website.
* Ridge regression has the highest R² (0.876). R² on the rupee scale is driven by a handful of luxury cars: the 10 worst-predicted test cars account for 76% of gradient boosting's squared error (81% for the random forest, 63% for ridge). Trees cannot extrapolate above the prices they have seen; a linear model on log price can. The single worst case is a 2020 BMW X7 listed at ₹72 lakh.
* The random forest is not better than gradient boosting on any metric and has a lower R² than ridge.
* Holding out whole listing titles (so no variant of a car is seen in training) gives gradient boosting R² 0.836 and MAPE 17.1%, random forest R² 0.819 and MAPE 18.0%.
* 90% conformal intervals covered **90.6%** of test prices; the median interval is 63% of the predicted price wide.

Permutation importance for gradient boosting (test R² drop): year 0.40, max power 0.37, model
0.16, brand 0.08, transmission 0.05, engine 0.05, km driven 0.05; fuel, owner, seats, seller
type and mileage are each below 0.01. Full tables are in [`reports/results.md`](reports/results.md)
and [`reports/metrics.json`](reports/metrics.json).

## 5. Web app

`web/` is a static site on Vercel.

* A visitor picks brand and model (dropdowns built from the training data, with listing counts), year, km, fuel, transmission, owner count and seller type; specifications are pre-filled with the median for that model and fuel and can be edited.
* The gradient-boosting model and the two conformal quantile models are exported to JSON (`web/data/model.json`, about 1.9 MB, 0.7 MB gzipped) and evaluated in the browser by `web/predict.js`, a 60-line tree walker. There is no server-side inference.
* The page warns when a model has fewer than 10 listings or the chosen year is outside the model's range in the data.
* The "Market trends" section (Chart.js) shows median price by model year and by kilometre band, split by fuel type, with a transmission filter, log/linear axis toggle and tooltips with counts and interquartile ranges. It replaces the Tableau dashboard mentioned in older descriptions; no Tableau work exists for this project.

**Parity with Python.** At export time the JSON trees are re-evaluated in Python for all 1,386
test cars and compared with scikit-learn: the maximum relative difference is 0. The Node test
evaluates `predict.js` on 300 stored scikit-learn predictions: maximum relative difference
2e-16. The deployed page was checked with Playwright on three cars against the Python
evaluator.

## 6. Limitations

* The data is a 2023 snapshot of CarDekho asking prices, newest model year 2020. It is not a current market price and not a transaction price.
* Condition, accident history, city, colour and exact trim are missing, which puts a floor on achievable error.
* Luxury and rare models have few listings and are usually under-predicted; see the error-share figure above.
* Brand and model are parsed from the listing title with simple rules, so a few models are merged or split imperfectly.
* The random split can place near-identical listings (same car, slightly different km) in train and test; the grouped-title split above is the more conservative estimate.
* The interval's 90% coverage is a marginal guarantee over cars like those in the data, not per car.

## Repository layout

```
notebooks/original/   the 2022 notebook, unchanged
data/raw/             Kaggle CSVs (ODbL / DbCL)
src/carprice/         data cleaning, models, metrics, conformal intervals, export, audit, CLI
tests/                pytest suite and Node parity test
reports/              audit.json, metrics.json, results.md
web/                  static site deployed to Vercel
```
