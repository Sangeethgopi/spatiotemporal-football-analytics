# REAL BASELINE RESULTS — Final Test Evaluation

## Design
- **TRAIN:** Metrica Match 2 (real, full — 141,156 frames, 78 turnover events)
- **VAL:** Metrica Match 3, first 70% minus purge (100,532 frames)
- **TEST:** Metrica Match 3, final 30% — **untouched until this evaluation** (43,129 frames)
- **Model:** HistGradientBoosting (learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5)
- **Smoothing:** 3.0 s causal (75-frame backward rolling mean, center=False)
- **Threshold:** 0.15 (selected from validation only)

---

## Test Results at Frozen Threshold (0.15)

| Metric | Value |
| :--- | :--- |
| **Event F1** | **0.604** |
| Event Precision | 0.533 |
| Event Recall | 0.696 |
| **TP** | **16** |
| **FP** | **14** |
| **FN** | **7** |
| GT Events | 23 |
| Predicted Events | 34 |
| ROC-AUC | 0.837 |
| PR-AUC | 0.238 |
| Bootstrap 95% CI (Event F1) | [0.490, 0.702] |

---

## At Threshold 0.50 (for comparability with synthetic experiment — diagnostic only)

| Metric | Value |
| :--- | :--- |
| Event F1 | 0.083 |
| TP | 1 |
| FP | 0 |
| FN | 22 |

> [!WARNING]
> The threshold 0.50 was selected on synthetic data. On real data, the probability distribution is compressed and this threshold is inappropriate. The above result at 0.50 is provided for transparency only; it must not be compared directly with the synthetic 0.600 result.

---

## Uncertainty

With 23 GT events in the test set, point estimates carry substantial uncertainty.

- Point estimate: **Event F1 = 0.604**
- Bootstrap 95% CI: **[0.490, 0.702]**

The CI is narrower than the synthetic experiment's [0.385, 0.765] because there are more GT events (23 vs 19) and fewer FPs distorting precision.
