# REAL MODEL CONFIGURATION AUDIT

## 1. Model Configuration
The final frozen model evaluated on the real test set was:
- **Class**: `HistGradientBoostingClassifier`
- **learning_rate**: `0.04`
- **max_leaf_nodes**: `15`
- **l2_regularization**: `1.5`
- **random_state**: `42`
- **class_weighting**: `None`

## 2. Configuration Freezing
This exact configuration was imported without modification from the preceding Synthetic model improvement protocol. **No hyperparameter tuning was performed on the real test set.**

## 3. Class Weighting Verification
The validation sweep tested a `HistGB_balanced` variant, but it degraded Event F1 (from 0.618 down to 0.389). The standard unweighted configuration was frozen.

## 4. Calibration Audit
No post-hoc calibration wrapper (e.g., `CalibratedClassifierCV`) was used in the real-to-real script. The model outputs raw `.predict_proba()` scores directly to the temporal smoothing layer. 

## 5. Conclusion
**SAFE.** The model configuration was properly frozen. No hyperparameters or calibration mechanisms accessed the test data.
