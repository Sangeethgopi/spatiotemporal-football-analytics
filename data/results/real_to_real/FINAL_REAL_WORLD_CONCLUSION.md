# FINAL REAL-WORLD CONCLUSION

## The Central Research Question

> *Can a model trained on real professional football tracking data generalise to an unseen real football match for turnover-associated press-trigger detection?*

**Answer: Yes, with appropriate threshold selection and causal temporal smoothing.**

---

## Bottleneck Classification

**A. Synthetic-data mismatch** — **confirmed and resolved**

The synthetic-to-real transfer failure was entirely attributable to probability calibration mismatch: the synthetic generator produced deterministic pressing signatures that led the model to output probability distributions incompatible with real football. When trained on real data, this failure disappears.

**B. Tracking/perception quality** — **partially resolved**

Professional optical tracking (Metrica, 59–68% ball coverage, median run 338 frames) provides adequate ball continuity for the downstream pipeline. Broadcast-CV tracking (26.5% coverage, median run 2 frames) is insufficient. This remains a limitation for deployment from broadcast video.

**C. Target/label formulation** — **acceptable but sensitive**

The 3m proximity threshold produces 78–102 events per ~95-minute match on real data. The operational target is reproducible and sufficient. It remains a weakly-supervised proxy; the 3m spatial threshold is arbitrary but robust (4m produces similar quantities).

**D. Feature representation** — **adequate**

ROC-AUC = 0.837 on the real test set confirms meaningful discriminative signal in the spatial/kinematic features derived from real tracking.

**E. Model capacity** — **not the bottleneck**

HistGradientBoosting consistently outperforms Logistic Regression and Random Forest on both synthetic and real validation data. The performance gap between models is smaller than the gap caused by smoothing window selection.

---

## What is Limiting Real-World Event F1?

The real test Event F1 = 0.604 breaks down as TP=16, FP=14, FN=7. The principal sources of limitation are:

1. **14 False Positives** — indicating spurious spatial pressure patterns that do not correspond to qualifying turnovers. Likely causes: incomplete possession assignment, possessions below the 3m threshold, and the inherent imprecision of proximity-based target construction.

2. **7 False Negatives** — events the model misses. Likely causes: gradual possession transitions without sharp pressure geometry, ball gaps during the transition frame preventing proximity evaluation, and noisy features at the moment of turnover.

---

## What Should the Dissertation Use as Its Primary Result?

**The real-to-real evaluation is the scientifically primary result:**

| | |
| :--- | :--- |
| **Train** | Metrica Match 2 (real professional tracking) |
| **Test** | Metrica Match 3, final 30% (held out) |
| **Event F1** | **0.604** |
| **95% CI** | **[0.490, 0.702]** |
| **Precision** | 0.533 |
| **Recall** | 0.696 |
| **ROC-AUC** | 0.837 |
| **Threshold** | 0.15 (selected on real validation only) |
| **Smoothing** | 3.0 s causal |

The synthetic 0.600 result should be reported as a development baseline, with explicit acknowledgment that it was obtained on algorithmically-generated data.

---

## What Should NOT Be Claimed

| ❌ Do NOT claim | ✅ Instead write |
| :--- | :--- |
| "Event F1 = 0.600 on football tracking data" | "Event F1 = 0.600 on synthetic development data; 0.604 on real held-out data" |
| "The model predicts pressing 4 seconds ahead" | "Positive labels are projected 100 frames backward from turnover events" |
| "Detects genuine pressing intent" | "Detects an operational proxy for turnover-associated press-trigger states" |
| "Airtight generalisation" | "Generalisation to an unseen match with appropriate threshold re-selection" |

---

## Final Experiments to Stop

1. ❌ Further synthetic data optimisation — the synthetic generator is not representative of real football.
2. ❌ Architecture exploration on synthetic data — model capacity is not limiting real-data performance.
3. ❌ Smoothing window hunting on the test set — configuration is now frozen.
4. ❌ Broadcast-CV improvements without first achieving stable possession assignment.

---

## Recommended Dissertation Framing

The dissertation tells a coherent, scientifically honest story:

1. A complete press-trigger detection pipeline was designed and evaluated on synthetic simulation data (Event F1 = 0.600, threshold 0.50).
2. External validation on real tracking revealed severe domain shift: the synthetic threshold never triggered on real data.
3. A real-to-real experiment (Match 2 → Match 3) demonstrated that the pipeline generalises to real football when the threshold is appropriately re-selected on real validation data (Event F1 = 0.604, threshold 0.15, 95% CI [0.490, 0.702]).
4. Broadcast-CV perception remains the barrier to deployment from video, where ball continuity (median 2 frames) prevents stable possession and turnover derivation.
