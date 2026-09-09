# REAL FEATURE LEAKAGE AUDIT

## 1. Feature Extraction Method
Features were extracted using `pd.Series.rolling(W, center=False)` and `.diff()`. The `center=False` argument ensures the rolling window relies only on `[t - W + 1, t]`, meaning all rolling features are strictly causal. 

## 2. Feature Causality Check
- **Base Spatial Features**: Current frame `t` only (Causal).
- **Kinematic Diff Features**: `pd.Series.diff(1)` and `diff(3)` (Causal).
- **Rolling Features**: `W=25`, `center=False` (Causal).
- **Temporal Derivatives**: `pd.Series.diff(25)` (Causal).

## 3. Boundary Feature Bleed Check
Because rolling features are computed across the full Match 3 DataFrame before splitting, the first 25 frames of the TEST set use historical data spanning backward into index 100,608. 
- **Does this cause future leakage?** No. Test data never bleeds backward into training/validation.
- **Does this cause training data to leak into test evaluation?** No. The frames `100,608` to `100,631` belong to the 100-frame **purge zone**, which was strictly excluded from the validation set. 
- **Conclusion**: The boundary effect is mathematically sound. The test set borrows history from the safely discarded purge zone, ensuring no data used to fit/select the model influences the test evaluation.

## 4. Normalisation
No full-match statistical normalisation (e.g., standard scaling) was applied before the split. 

## 5. Conclusion
**SAFE.** All features are strictly causal. No future information, and no training information, leaks into the test set feature representation.
