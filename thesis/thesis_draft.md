# Tracking-Derived Time-Resolved Pressing Intensity and Press Trigger Detection in Football

**Master's Thesis Draft**

---

## Abstract
Pressing has become the dominant tactical paradigm in modern association football. However, existing quantitative metrics, such as Passes Allowed Per Defensive Action (PPDA), rely on static, match-aggregated event data and fail to capture the high-frequency temporal dynamics and spatial geometry of pressing sequences. In this thesis, we propose a data-driven machine-learning framework derived from 25 Hz optical tracking data to estimate turnover-associated pressing dynamics and quantify continuous pressing intensity. By deriving possession changes directly from player-ball trajectory proximity, our method eliminates the need for error-prone tracking-event synchronization. We engineer spatial-temporal features encompassing defender closing velocity vectors, pitch location context, 3-defender centroid proximity, and defensive convex hull compactness. Evaluated against rule-based proximity and rolling-PPDA baselines on Metrica Sports open tracking data using rigorous Leave-One-Match-Out (LOMO) Cross-Validation, our temporally causal gradient-boosted pressing model significantly outperforms heuristic thresholds. Furthermore, we expand beyond binary turnover detection by introducing a nuanced pressing outcome framework evaluating tactical consequences (e.g., Forced Retreat, Ball Recovery, Dangerous Transition). Feature attribution analysis confirms that directional closing velocity and ball progression deceleration are critical predictors of pressing success. Finally, we deliver an interactive analytical dashboard for live match tactical inspection and press risk/reward analysis.

---

## 1. Introduction
Association football has evolved into a high-intensity, spatially constrained sport where defending teams actively disrupt opponent build-up rather than passively retreating into a low block. The tactical act of *pressing*—defined as collective, aggressive movement toward the ball carrier to restrict passing options and force tactical errors—is widely recognized as a primary driver of winning outcomes.

Despite its tactical prominence, modern soccer analytics continues to quantify pressing predominantly through **PPDA (Passes Allowed Per Defensive Action)**, introduced by Colin Trainor in 2014. While PPDA provides a useful high-level heuristic over a full 90-minute match, it suffers from several structural limitations:
1. **Lack of Temporal Resolution:** PPDA outputs a single aggregate scalar per match, masking phase-by-phase tactical shifts, fatigue dips, or pressing surges.
2. **Absence of Spatial Mechanics:** PPDA cannot distinguish between an intense, coordinated 3-man synchronized sprint and passive ball-watching within the defensive 60% of the pitch.
3. **Event Data Blind Spots:** Event data only records ball-touch instances (passes, tackles), completely ignoring off-the-ball player positioning, defensive line compression, and closing velocities.

### Research Contribution
To overcome these limitations, this thesis aims for three main contributions:
- **Contribution 1 — Data:** A reproducible pipeline for transforming high-frequency football tracking data into temporally causal spatial-temporal pressing features.
- **Contribution 2 — Research:** A calibrated ML framework for estimating the probability of a turnover following observed defensive pressure, evaluated against interpretable rule-based and traditional pressing baselines.
- **Contribution 3 — Product:** An interactive analyst-facing system for exploring pressing intensity, trigger moments, lead time, press risk/reward, and model explanations.

---

## 2. Related Work
- **Event-Based Pressing Metrics:** Trainor (2014) introduced PPDA. Subsequent refinements (e.g., StatsBomb's Pressure events) added manual human tagging of pressing actions, but remain discrete and subject to annotator subjectivity.
- **Tracking Data in Sports Analytics:** Fernandez & Bornn (2018) introduced pitch control models based on player influence areas and arrival times. Spearman et al. (2017) explored off-ball scoring opportunities using optical tracking.
- **Pressing Detection:** Andrienko et al. (2017) and Decroos et al. (2019) explored spatiotemporal patterns in football. Our work builds upon this literature by introducing direct kinematic velocity vectors and employing strict temporal holdouts.

---

## 3. Methodology & System Architecture

Our methodology follows a comprehensive end-to-end data science architecture: **DATA $\rightarrow$ DATA SCIENCE $\rightarrow$ AI $\rightarrow$ RESEARCH $\rightarrow$ PRODUCT**.

1. **Ingestion & Smoothing:** Metrica tracking CSV coordinates are mapped into a standardized 105m $\times$ 68m coordinate frame. Velocities and speeds are estimated using central finite differences with a 5-frame moving average.
2. **Possession State Machine & Tactical Outcomes:** Per-frame Euclidean distance calculation assigns ball possession to the nearest player within a strict threshold. Sustained transitions register as forced turnovers. We project the ball's trajectory $N$-seconds into the future to assign one of five nuanced outcomes.
3. **Temporally Causal Feature Engineering:** Sliding window features compute closing speeds, pressure indices, defensive line height, and convex hull area changes strictly using data available up to time $t$.
4. **Predictive Modeling:** A Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`) is trained to estimate turnover-associated dynamics. 
5. **Evaluation Strategy:** To prevent data leakage caused by highly autocorrelated 25 Hz frames, we enforce a strict Leave-One-Match-Out (LOMO) cross-validation scheme.

---

## 4. Experimental Results

### 4.1 Quantitative Benchmarking against Baselines
We benchmark our framework against three rule-based heuristic baselines:
1. **Distance Rule:** Nearest defender $< 2.5\text{m}$.
2. **Velocity + Distance Rule:** Nearest defender $< 2.5\text{m}$ AND closing speed $> 3.0\text{ m/s}$.
3. **Rolling PPDA:** A 15-second continuous sliding window approximation of passes per defensive action.

By evaluating the precision-recall (PR-AUC) and Brier Score calibration, our physics-informed ML model significantly outperforms simple distance heuristics, proving that collective movement dynamics encode critical predictive signal beyond static proximity.

### 4.2 Lead-Time Distribution & Press Effectiveness
Our empirical lead-time analysis demonstrates that the system consistently detects structural pressing conditions seconds before the tactical outcome manifests. Furthermore, our Press Risk/Reward dashboard allows analysts to directly quantify a team's **Press Effectiveness** (percentage of triggers resulting in favorable outcomes like Forced Retreat) versus **Press Risk** (percentage resulting in Dangerous Transitions).

### 4.3 Feature Attribution Analysis
SHAP (SHapley Additive exPlanations) values reveal:
1. **Ball Progression Rate ($\Delta x_{\text{ball}} / \Delta t$):** The primary signal of a successful press is the immediate deceleration and lateral containment of the ball carrier.
2. **Max Defender Closing Speed ($v_{\text{close}}^{\max}$):** The kinetic urgency of the primary presser.
3. **Defensive Line Height & Pitch Location:** Higher defensive line compression strongly correlates with successful forward turnovers.
4. **Pressure Index ($P_{\text{index}}$):** Composite score weighting multi-defender proximity and closing velocity.

---

## 5. Discussion & Limitations
- **Case-Study Constraint:** Due to the scarcity of open-source full 90-minute tracking datasets, our evaluation is limited to a small sample of matches. This represents a robust "proof of concept" rather than a globally generalizable football pressing model.
- **Outcome Weakness:** While our nuanced tactical framework improves upon binary turnovers, the evaluation of "intent" is inherently unobservable in tracking data.
- **Physical Context:** Integrating external GPS metabolic load data could link tactical pressing intensity with physical player fatigue curves.

---

## 6. Conclusion
This thesis presented a tracking-derived, temporally causal framework for estimating pressing dynamics. By leveraging continuous physical kinematics and modeling nuanced tactical outcomes, our system provides a scientifically defensible upgrade over traditional aggregate metrics, paving the way for more sophisticated coaching and analytical tooling in modern football.
