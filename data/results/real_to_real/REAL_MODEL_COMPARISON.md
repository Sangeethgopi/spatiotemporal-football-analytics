# REAL MODEL COMPARISON — Validation Results

## Setup
All models were trained on Match 2 (full) and evaluated on Match 3 first 70%, using the frozen smoothing configuration (3.0 s, threshold 0.15).

| Model | Val Event F1 | Val ROC-AUC | Notes |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.502 | 0.801 | Convergence warning (1000 iterations insufficient) |
| Random Forest | 0.461 | 0.827 | Slightly lower F1 despite competitive ROC |
| **HistGradientBoosting** | **0.618** | **0.825** | **Best — frozen** |
| HistGB (balanced class weights) | 0.389 | 0.826 | Class weighting hurts event F1 |

## Conclusions

1. **HistGradientBoosting remains the best model** on real data, consistent with the synthetic evaluation.
2. **Model capacity is NOT the bottleneck** — all models achieve meaningful validation F1, and the ordering is consistent with the synthetic experiment.
3. **ROC-AUC is similar across all models** (0.801–0.827), confirming that discriminative signal exists in the features regardless of model architecture.
4. **Balanced class weights reduced event F1**, likely because the aggressive over-weighting of the minority class increased false positives under the strict evaluator.
