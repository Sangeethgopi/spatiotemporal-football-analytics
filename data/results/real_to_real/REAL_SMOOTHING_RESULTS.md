# REAL SMOOTHING RESULTS — Validation-Only Selection

## Overview
Smoothing window and threshold were selected **exclusively using the validation set** (Match 3 first 70%, minus 100-frame purge). The test set was never consulted.

## Smoothing Sensitivity (Validation)

| Smooth Window | Best Val Event F1 | Best Val Threshold | Notes |
| :--- | :--- | :--- | :--- |
| 0 s (none) | 0.366 | 0.50 | High fragmentation — same pattern as synthetic experiment |
| 0.5 s (12 fr) | 0.496 | 0.40 | Substantial improvement |
| 1.0 s (25 fr) | 0.515 | 0.25 | Used in prior synthetic experiment |
| 1.5 s (37 fr) | 0.562 | 0.25 | Continued improvement |
| 2.0 s (50 fr) | 0.591 | 0.20 | Near plateau |
| **3.0 s (75 fr)** | **0.618** | **0.15** | **Best validation F1 — frozen** |

**Frozen configuration: smoothing = 3.0 s, threshold = 0.15**

## Key Observations

1. **Smoothing is necessary on real data**, just as it was on synthetic data. Unsmoothed predictions fragment severely (validation F1 = 0.366).

2. **The optimal threshold on real data (0.15) is fundamentally different from the synthetic threshold (0.50).** This confirms the calibration shift identified in the domain shift analysis. Probabilities are compressed on real data; forcing threshold = 0.50 would have yielded F1 = 0.083 on the test set.

3. **A longer smoothing window (3.0 s) is needed for real data than for synthetic (1.0 s).** Real football pressing signals are noisier and require more temporal integration to form stable predictions.

4. **The model shows a consistent monotonic improvement with longer smoothing on validation**, suggesting that real pressing-associated features build gradually over several seconds rather than spiking instantaneously.
