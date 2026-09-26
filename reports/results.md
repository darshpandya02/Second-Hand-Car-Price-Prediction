| Model | Test R² [95% CI] | MAE, ₹ lakh [95% CI] | MAPE [95% CI] | RMSE, ₹ lakh [95% CI] | CV R² (5-fold) |
|---|---|---|---|---|---|
| Median baseline | -0.046 [-0.064, -0.032] | 2.60 [2.40, 2.81] | 79.0% [73.0%, 85.8%] | 4.79 [3.84, 5.79] | -0.052 ± 0.013 |
| Linear (ridge, log price) | 0.876 [0.810, 0.927] | 0.75 [0.68, 0.83] | 16.6% [15.7%, 17.5%] | 1.65 [1.16, 2.15] | 0.897 ± 0.009 |
| Random forest | 0.799 [0.704, 0.925] | 0.75 [0.66, 0.86] | 16.5% [15.6%, 17.5%] | 2.10 [1.07, 2.96] | 0.846 ± 0.070 |
| Gradient boosting (HGB) | 0.837 [0.768, 0.929] | 0.73 [0.65, 0.82] | 15.7% [14.9%, 16.7%] | 1.89 [1.05, 2.60] | 0.860 ± 0.076 |

Share of test squared error from the 10 worst-predicted cars: Median baseline: 52%, Linear (ridge, log price): 63%, Random forest: 81%, Gradient boosting (HGB): 76%.

Grouped split (whole listing titles held out):

| Model | R² | MAE, ₹ lakh | MAPE | RMSE, ₹ lakh |
|---|---|---|---|---|
| Random forest | 0.819 | 0.87 | 18.0% | 1.94 |
| Gradient boosting (HGB) | 0.836 | 0.84 | 17.1% | 1.84 |

90% conformal interval: test coverage 90.6%, median width 63% of the predicted price.

Permutation importance (test R² drop, gradient boosting):

| Feature | Mean drop | Std |
|---|---|---|
| year | 0.404 | 0.057 |
| max_power | 0.373 | 0.033 |
| model | 0.163 | 0.037 |
| brand | 0.084 | 0.011 |
| transmission | 0.053 | 0.010 |
| engine | 0.048 | 0.010 |
| km_driven | 0.046 | 0.015 |
| fuel | 0.006 | 0.002 |
| owner | 0.006 | 0.004 |
| seats | 0.003 | 0.002 |
| seller_type | 0.001 | 0.004 |
| mileage | -0.004 | 0.013 |
