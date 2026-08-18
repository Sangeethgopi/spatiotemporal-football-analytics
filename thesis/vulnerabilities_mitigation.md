# Production Mitigation Strategies for Real-World Tracking Data

This document details the mathematical formulations and architectural blueprints designed to address the four primary vulnerabilities of deploying pressing intensity models on raw, un-sanitized match tracking feeds.

---

## 1. Kinematic Level: Kalman Filtering with Constant-Velocity Coasting

### The Problem
Optical tracking coordinates suffer from camera occlusions, causing player coordinates to drop out or "jump" instantly across the pitch. Simple backward finite differences translate this coordinate jump into physically impossible velocity spikes (e.g., >50 m/s), which corrupts features like closing speed and defensive convex hull area.

### The Algorithmic Patch
We implement a Constant Velocity Kalman Filter to estimate smooth player state trajectories.

#### State Vector
$$\mathbf{x}_t = \begin{bmatrix} x_t & y_t & v_{x, t} & v_{y, t} \end{bmatrix}^T$$

#### State Transition Model
$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q})$$

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

#### Measurement Model
$$\mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, \mathbf{R})$$

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

#### Anomaly Rejection and Coasting
Before updating the filter covariance, we calculate the Mahalanobis distance of the innovation vector $\mathbf{y}_t = \mathbf{z}_t - \mathbf{H}\mathbf{x}_{t|t-1}$:

$$d^2 = \mathbf{y}_t^T \mathbf{S}_t^{-1} \mathbf{y}_t, \quad \mathbf{S}_t = \mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^T + \mathbf{R}$$

If $d^2 > \chi^2_{\alpha, 2}$ (the tracking coordinate has jumped beyond physical sprint capabilities):
1. The measurement is rejected.
2. The state is updated using prediction equations only (momentum-based coasting):
   $$\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1}$$
   $$\mathbf{P}_{t|t} = \mathbf{P}_{t|t-1}$$

---

## 2. Possession Level: persistence-buffered State Machine with Heading Alignment

### The Problem
When players shield the ball or contest it in tight zones, 2D proximity filters flip the possession state rapidly between teams. This creates high-frequency turnover noise, causing the target label $y$ to oscillate.

### The Algorithmic Patch
We upgrade the 3-State Possession FSM with a temporal persistence buffer and movement heading alignment.

```
                  POSSESSION STATE TRANSITION MODEL
                  
       ┌───────────────────► [ Flight / Loose ] ◄──────────────────┐
       │                            │                              │
       │  Ball proximity > 2.0m     │ Proximity <= 2.0m            │ Ball proximity > 2.0m
       │                            ▼                              │
 [ Team A Control ] ◄────────────────────────────────────────► [ Team B Control ]
                     Proximity <= 2.0m for >= 5 frames (0.2s)
                     AND Alignment Vector cos(θ) >= 0.0
```

#### Heading Vector Alignment
The heading vector of the player $\mathbf{d}_{\text{player}}$ and the velocity vector of the ball $\mathbf{v}_{\text{ball}}$ are compared using a dot product:

$$\cos(\theta) = \frac{\mathbf{d}_{\text{player}} \cdot \mathbf{v}_{\text{ball}}}{\|\mathbf{d}_{\text{player}}\|_2 \|\mathbf{v}_{\text{ball}}\|_2}$$

- If $\cos(\theta) \ge 0.0$: The player is moving in structural alignment with the ball (shielding or dribbling). Possession is maintained.
- If $\cos(\theta) < 0.0$ and another player is closer with $\cos(\theta) \ge 0.0$: Possession switch is initiated.

#### Temporal Buffer
A possession switch from Team A to Team B is only registered if the proximity condition ($d \le 2.0\text{m}$) and alignment condition are sustained for at least $N = 5\text{ frames}$ ($0.2\text{ seconds}$).

---

## 3. Feature Level: Multi-Horizon Predictors for Fast Counters

### The Problem
During rapid transition phases (e.g., long counter-attacking balls), the spatial layout of the match shifts in milliseconds. A fixed 1.5s or 3.0s lookahead prediction becomes lagging and reactive during sudden counter-attacks.

### The Algorithmic Patch
We replace single-target training with a multi-horizon predictor outputting risk curves.

#### Output Vector
The target vector $\mathbf{y}_t$ is structured as three distinct prediction targets:

$$\mathbf{y}_t = \begin{bmatrix} y_{t+5} & y_{t+25} & y_{t+75} \end{bmatrix}$$

- $y_{t+5}$ ($0.2\text{s}$ - Immediate transition risk - counter-attack alert)
- $y_{t+25}$ ($1.0\text{s}$ - Tactical pressing transition risk)
- $y_{t+75}$ ($3.0\text{s}$ - Strategic territorial pressing risk)

#### Classifier Adaptation
The model trains using a multi-output classifier to output a continuous risk trajectory:

```python
# Multi-output prediction setup
from sklearn.multioutput import MultiOutputClassifier

base_clf = HistGradientBoostingClassifier(random_state=42)
multi_clf = MultiOutputClassifier(base_clf)
multi_clf.fit(X_train, np.column_stack([y_02s, y_10s, y_30s]))
```

---

## 4. Probability Level: Context-Aware Isotonic Override

### The Problem
Globally calibrated models underperform in extreme, highly obvious scenarios (e.g., an open-goal tap-in) because global Platt/Sigmoid calibration limits probabilities to $[0.05, 0.90]$ to account for general tracking noise.

### The Algorithmic Patch
We apply a two-stage calibration pipeline: global sigmoid calibration for general sequences, and a rule-based isotonic override for extreme context events.

#### Override Condition
Let the ball carrier coordinate be $(x_{\text{ball}}, y_{\text{ball}})$ and the goal mouth be $(105.0, 34.0)$:

1. **Spatial Proximity:** The ball carrier is within the penalty area:
   $$x_{\text{ball}} \ge 88.5\text{ m}, \quad 13.84\text{ m} \le y_{\text{ball}} \le 54.16\text{ m}$$
2. **Defensive Clearances:** The nearest defender distance $d_1 > 5.0\text{ m}$ (no physical pressure).
3. **Goalkeeper Positioning:** The goalkeeper is out of the goal mouth line.

If all three conditions are satisfied, the global probability calibrator is bypassed:

$$P(\text{Success}) \leftarrow \max\big(P(\text{Success}), 0.99\big)$$

---

## 5. Tactical Level: Modeling Pressing during Aerial Trajectories (Lofted Passes / Air Balls)

### The Challenge
When a long lofted pass or cross is played, the ball travels high above the playing pitch. Physically, defenders cannot directly press or challenge the ball *while it is in flight*. During this period, there is no physical "ball carrier" to put under pressure.

### How the Framework Mathematically Models Aerial Phases

#### 1. FSM Transition to "Contested / In-Flight" State
When a lofted pass is made, the distance between the ball and all players exceeds the control threshold ($2.0\text{m}$):
$$\min_{j} \|\mathbf{p}_j(t) - \mathbf{p}_{\text{ball}}(t)\|_2 > 2.0\text{ m}$$

In [possession.py](file:///C:/Users/SANGEETH%20T/.gemini/antigravity-ide/scratch/pressing_intensity_thesis/src/possession.py), the possession state machine immediately transitions the frame's raw state to **`Contested / In-Flight`**. The tactical possession remains with the passing team, but the ball carrier status is set to `None`.

#### 2. Pressure Decay via Spatial Expansion
Because the ball is in flight, the distance coordinates $d_1(t), d_2(t), d_3(t)$ relative to the ball expand rapidly (often to $15.0\text{m} - 30.0\text{m}$). 

The distance decay factor in the Physical Pressure Index:
$$w_{\text{dist}} = \frac{1}{\max(d_k(t), 1.0)}$$

For a ball in flight ($d_k \approx 20\text{m}$), $w_{\text{dist}} \approx \frac{1}{20.0} = 0.05$.
Consequently, the **Pressure Index ($P_{\text{index}}$) decays to near-zero ($<0.05$)** for the duration of the ball's flight.

#### 3. Receiving-Point Pressing Spikes (Tactical Reality)
As the ball descends toward the receiving player $r$:
- The receiver's distance to the ball drops below $2.0\text{m}$, re-engaging ball carrier calculations.
- Defending players converge on the landing zone, causing their **closing velocity vector ($v_{\text{close}}$)** to spike as they rush to contest the first touch.
- This creates a **sharp pressing urgency peak ($P(\text{Trigger}) \ge 0.70$)** precisely at the coordinate where the ball lands.

This formulation matches tactical reality: pressing is inactive while the ball is traveling in the air, but intensifies immediately upon receipt.

