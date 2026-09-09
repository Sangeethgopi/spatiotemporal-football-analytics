# REAL EVENT EVALUATION AUDIT

## 1. Evaluator Implementation
The script imports `evaluate_events_strict` from `run_apples_apples.py`. This is the identical strict evaluator used to audit the synthetic models, ensuring perfect internal comparability across the dissertation.
- `merge_gap = 0` (a single frame dropping below the threshold splits an event).
- Matching tolerance: 50 frames (2.0s) expansion of the 100-frame GT horizon.

## 2. True Positives vs Predicted Events Math
The audit verified the following counts on the TEST set:
- **GT Events:** 23
- **Predicted Events:** 34
- **True Positives (TP):** 16 (matched GT events)
- **False Positives (FP):** 14 (unmatched predicted events)
- **False Negatives (FN):** 7 (unmatched GT events)

### Why doesn't TP + FP = Predicted Events?
`16 (TP) + 14 (FP) = 30 ≠ 34 (Predicted)`

This occurs because the evaluator defines **TP as the number of Ground Truth (GT) events matched**, not the number of predicted events matched. 
In the test set, 20 predicted events successfully overlapped with GT events. Because predictions can fragment (even with 3.0s smoothing), those 20 predicted events covered exactly 16 GT events. 

### Precision Calculation Consequence
The evaluator calculates precision as:
`Precision = TP / (TP + FP) = 16 / (16 + 14) = 16 / 30 = 0.533`

A strict prediction-centric precision would be `20 / 34 = 0.588`. By using GT events in the numerator and unmatched predictions in the denominator, the evaluator implements a **hybrid metric** that slightly penalises prediction fragmentation.

## 3. Conclusion
**SAFE, WITH NOTE.** The mathematical accounting is verified and precise. While the precision calculation uses a hybrid formula, this identical evaluator was used across the entire study (Synthetic Baseline, Synthetic Candidate, and Real-to-Real), meaning all comparisons remain scientifically valid and internally consistent.
