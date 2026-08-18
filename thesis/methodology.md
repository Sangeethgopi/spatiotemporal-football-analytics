# Formal Methodology & Mathematical Formulations (Validated)

## 1. Physical Kinematic Bounds & Strictly Causal Velocity Estimation

Let entity $i$ have raw tracking coordinates $(\tilde{x}_i(t), \tilde{y}_i(t)) \in [0, 1]^2$ sampled at frequency $f_s = 25\text{ Hz}$ ($\Delta t = 0.04\text{ s}$).
Metric pitch coordinates $(x_i(t), y_i(t))$ on a pitch of length $L = 105.0\text{ m}$ and width $W = 68.0\text{ m}$ are:

$$x_i(t) = \tilde{x}_i(t) \cdot L, \quad y_i(t) = \tilde{y}_i(t) \cdot W$$

### A. Strictly Causal Finite Differences (Zero Future-Leakage)
To completely prevent future information leakage into time $t$, velocity vectors $\mathbf{v}_i(t) = (v_{x, i}(t), v_{y, i}(t))$ are computed using backward finite differences with a 5-frame causal backward rolling average:

$$\mathbf{v}_i^{\text{raw}}(t) = \frac{\mathbf{p}_i(t) - \mathbf{p}_i(t - 1)}{\Delta t}$$

$$\mathbf{v}_i(t) = \frac{1}{K} \sum_{k=0}^{K-1} \mathbf{v}_i^{\text{raw}}(t - k), \quad K = 5$$

### B. Physical Human Sprint Bound
Human biomechanics bounds maximum sustained sprinting speed:
$$\|\mathbf{v}_i(t)\|_2 \le 10.5\text{ m/s} \quad (\approx 37.8\text{ km/h}) \quad \forall i \in \text{Players}$$
$$\|\mathbf{v}_{\text{ball}}(t)\|_2 \le 36.0\text{ m/s} \quad (\approx 130\text{ km/h})$$

If raw tracking noise produces $\|\mathbf{v}_i(t)\|_2 > v_{\max}$, the vector is causally scaled:
$$\mathbf{v}_i(t) \leftarrow \mathbf{v}_i(t) \cdot \frac{v_{\max}}{\|\mathbf{v}_i(t)\|_2}$$

---

## 2. 3-State Possession Hysteresis & Turnover Derivation

To prevent artificial turnover inflation from noisy single-frame coordinate fluctuations, we define a 3-State Possession State Machine:
1. **Direct Ball Proximity:** Player $j$ controls the ball if $d_j(t) = \|\mathbf{p}_j(t) - \mathbf{p}_{\text{ball}}(t)\|_2 \le 2.0\text{ m}$.
2. **Neutral / In-Flight State:** If $\min_j d_j(t) > 2.0\text{ m}$, the state is marked as contested/in-flight (passes, loose clearances) and the prior team retains tactical control.
3. **Turnover Event ($t_{\text{to}}$):** A valid possession turnover is registered only when Team A has maintained settled control for $\ge 3.0\text{ s}$ (75 frames) and Team B wins and maintains controlled possession for $\ge 2.0\text{ s}$ (50 frames).

---

## 3. Spatial-Temporal Pressing Features

For defending player set $\mathcal{D}(t) = \{d_1, d_2, \dots, d_K\}$:

### A. Directional Closing Velocity ($v_{\text{close}}$)
The unit vector from defender $k$ to the ball carrier is:
$$\hat{\mathbf{u}}_k(t) = \frac{\mathbf{p}_{\text{ball}}(t) - \mathbf{p}_k(t)}{\|\mathbf{p}_{\text{ball}}(t) - \mathbf{p}_k(t)\|_2}$$

The scalar closing speed along the direct line of approach is:
$$v_{\text{close}, k}(t) = \mathbf{v}_k(t) \cdot \hat{\mathbf{u}}_k(t)$$
$$v_{\text{close}}^{\max}(t) = \max_{k \in \mathcal{D}(t)} \left( \max(0, v_{\text{close}, k}(t)) \right) \le 10.5\text{ m/s}$$

### B. Standardized Physical Pressure Index ($P_{\text{index}}$)
$$P_{\text{index}}^{\text{raw}}(t) = \sum_{k=1}^3 \frac{1}{\max(d_{(k)}(t), 1.0)} \cdot \left( \max(0, v_{\text{close}, (k)}(t)) + 0.5 \right)$$

$$P_{\text{index}}(t) = 1.0 - \exp\left(-\frac{P_{\text{index}}^{\text{raw}}(t)}{3.0}\right) \in [0.00, 1.00]$$

---

## 4. Benchmark Baseline vs Calibrated ML Model

### A. Spatial Proximity Heuristic Baseline
Predicts an active press trigger based solely on static 1st-defender distance in the opponent's defensive half ($x \le 63\text{m}$ for Home, $x \ge 42\text{m}$ for Away):
$$\text{Score}_{\text{spatial}}(t) = \frac{1}{1.0 + d_{(1)}(t) / 2.5}$$

### B. Calibrated Machine Learning Model (Ours)
A regularized Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`, $\lambda = 1.5$, $\text{max\_leaf\_nodes} = 15$) calibrated via Sigmoid `CalibratedClassifierCV` over a strictly causal 70/30 temporal holdout split.

---

## 5. Empirical Lead-Time Metric

For each validated turnover event at $t_{\text{to}}$, the preemption lead time is defined as:
$$\Delta t_{\text{lead}} = t_{\text{to}} - \min \left\{ t \in [t_{\text{to}} - 5.0\text{s}, t_{\text{to}}] \mid P(\text{Trigger} \mid \mathbf{x}_t) \ge \tau \right\}$$
Reported via **Median**, **Interquartile Range (IQR)**, and **Mean $\pm$ Std**.
