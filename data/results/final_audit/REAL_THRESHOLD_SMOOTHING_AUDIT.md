# REAL THRESHOLD AND SMOOTHING AUDIT

## 1. Smoothing Audit
- **Implementation**: `pd.Series(p_test_raw).rolling(75, min_periods=1, center=False).mean()`
- **Causality**: `center=False` ensures that probability at time `t` depends only on `[t-74, t]`.
- **Val/Test Bleed**: The smoothing operation is applied **only to the test slice** (`p_test_raw`). It does not roll over the validation-test boundary, ensuring perfect isolation.

## 2. Threshold and Window Selection
The selected configuration is:
- **Smoothing window**: 3.0 seconds (75 frames)
- **Threshold**: 0.15

This was derived exclusively by sweeping the validation set (Match 3 first 70%).

| Smoothing | Best Val F1 | Selected Threshold |
| :--- | :--- | :--- |
| 0 s | 0.366 | 0.50 |
| 0.5 s | 0.496 | 0.40 |
| 1.0 s | 0.515 | 0.25 |
| 1.5 s | 0.562 | 0.25 |
| 2.0 s | 0.591 | 0.20 |
| **3.0 s** | **0.618** | **0.15** |

## 3. Leakage Check
- **Was the threshold adjusted to improve TEST performance?** No. 0.15 was frozen before the test set was evaluated.
- **Was the test set evaluated multiple times?** No. It was evaluated exactly once under the frozen configuration.

## 4. Conclusion
**SAFE.** The post-processing hyperparameter selection strictly adhered to the validation boundary.
