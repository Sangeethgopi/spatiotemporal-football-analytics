# Research Plan: Pressing Intensity & Trigger Detection

## 1. Title
**Tracking-Derived Time-Resolved Pressing Intensity and Press Trigger Detection in Association Football: Benchmarking Beyond PPDA**

## 2. Research Questions
- **Primary Research Question (RQ1):** Can a time-resolved spatial-temporal machine learning model derived solely from 25 Hz player and ball tracking data detect turnover-inducing pressing triggers significantly more accurately than traditional PPDA-based proximity metrics?
- **Secondary Research Question (RQ2):** Which spatial-temporal features (e.g., closing velocity vectors, 3-defender proximity, team compactness convex hull, defensive line height) contribute most decisively to the success of a defensive press?
- **Practical Application Question (RQ3):** How can time-resolved pressing intensity profiles be operationalized into actionable tactical analytics for coaching and opposition analysis?

## 3. Formal Hypotheses
- **$H_1$ (Superiority over Aggregate Baselines):** Continuous spatial-temporal tracking features achieve a statistically significant improvement in $F_1$-score and PR-AUC over naive PPDA thresholding baselines when predicting possession turnovers within a $T = 4.0\text{s}$ horizon.
- **$H_2$ (Dynamic vs. Static Cues):** Dynamic kinetic features (closing velocity $-\frac{dd}{dt}$ and ball deceleration) provide higher predictive power than static geometric proximity alone.

## 4. Dataset Description
- **Dataset:** Metrica Sports Open Tracking Dataset (Matches 1, 2, and 3).
- **Sampling Frequency:** 25 Hz (25 frames per second).
- **Pitch Coordinate System:** Standard FIFA pitch geometry normalized to $[0, 105]\text{ m} \times [0, 68]\text{ m}$.
- **Entities Tracked:** 22 outfield/goalkeeper players $(x, y, v_x, v_y)$ and the Match Ball $(x, y, v_x, v_y)$.

## 5. Scope and Assumptions
1. **Self-Contained Turnover Labeling:** Avoids brittle alignment between external event logs and raw tracking data by deriving ground-truth possession changes directly from continuous ball-player Euclidean proximity.
2. **Weak Supervision Horizon:** Frames occurring within $T \in [2.0, 4.0]\text{ s}$ prior to a forced turnover are labeled as active press triggers ($y = 1$).
