# REAL TARGET AUDIT

## 1. Target Construction (`target_press_trigger`)
The target relies on four criteria:
1. Defending team regains possession
2. Possession is retained for >= 3.0s (75 frames)
3. Defender <= 3.0m from the ball at the transition frame
4. Target label is projected backward exactly 100 frames

## 2. Target "Future" Information
The target inherently uses **future information** (evaluating possession retention for 75 frames after the turnover). This is mathematically correct and necessary for a supervised target. 

## 3. Possession Causality
The algorithm `assign_possession()` uses a strict forward scan (`i` from `0` to `n`). At frame `t`, possession state depends only on frame `t` spatial proximities and the counter of previous consecutive holds. 
- **Does possession smooth backward?** No. 
- **Does future possession leak into features?** No. The target label (which uses the 75-frame future hold) is strictly separated from the feature dataframe. Features rely entirely on the causal frame state.

## 4. Uniformity
The target construction function is applied identically across Match 2 and Match 3. 

## 5. Conclusion
**SAFE.** The target formulation correctly represents the supervised goal without leaking future information into the predictive features.
