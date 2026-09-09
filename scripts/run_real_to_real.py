"""
REAL-TO-REAL EXPERIMENT — Metrica Sports Open Data
Trains on real Match 2, validates on first 70% of Match 3, tests on final 30% of Match 3.
Uses vectorized feature extraction for speed.

Design:
  TRAIN  : Metrica Match 2 (full)
  VAL    : Metrica Match 3, frames 1 to int(0.7 * n3) - 100 (purge gap)
  TEST   : Metrica Match 3, frames int(0.7 * n3) to end  [NEVER TOUCHED until final eval]
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from kloppy import metrica
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.abspath('.'))
from scripts.run_apples_apples import evaluate_events_strict

PITCH_L = 105.0
PITCH_W = 68.0
HZ = 25
PURGE = 100
PROX_M = 3.0
HOLD_FRAMES = 75
HORIZON = 100
OUT = "data/results/real_to_real"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. FAST DATA LOADER
# ============================================================

def load_match(match_id):
    """Load a Metrica match into numpy arrays (fast)."""
    t0 = time.time()
    print(f"  Loading Match {match_id}...")
    ds = metrica.load_open_data(match_id=match_id, sample_rate=None, limit=None)
    
    ball_x = []; ball_y = []
    home_x = []; home_y = []; away_x = []; away_y = []
    frame_ids = []
    
    for frame in ds.frames:
        frame_ids.append(frame.frame_id)
        if frame.ball_coordinates:
            ball_x.append(frame.ball_coordinates.x * PITCH_L)
            ball_y.append(frame.ball_coordinates.y * PITCH_W)
        else:
            ball_x.append(np.nan); ball_y.append(np.nan)
        
        hxs, hys, axs, ays = [], [], [], []
        for player, coord in frame.players_coordinates.items():
            if coord is None: continue
            if player.team.ground.value == "home":
                hxs.append(coord.x * PITCH_L); hys.append(coord.y * PITCH_W)
            else:
                axs.append(coord.x * PITCH_L); ays.append(coord.y * PITCH_W)
        home_x.append(hxs); home_y.append(hys)
        away_x.append(axs); away_y.append(ays)
    
    n = len(frame_ids)
    print(f"  Match {match_id}: {n} frames in {time.time()-t0:.1f}s")
    return {
        "n": n, "frame_ids": np.array(frame_ids),
        "ball_x": np.array(ball_x, dtype=float),
        "ball_y": np.array(ball_y, dtype=float),
        "home_x": home_x, "home_y": home_y,
        "away_x": away_x, "away_y": away_y,
    }

# ============================================================
# 2. POSSESSION ASSIGNMENT (vectorized where possible)
# ============================================================

def assign_possession(m, hold=HOLD_FRAMES):
    n = m["n"]
    bx = m["ball_x"]; by = m["ball_y"]
    poss = np.full(n, -1, dtype=int)  # -1=None, 0=Home, 1=Away
    cur = -1; consec = 0
    
    for i in range(n):
        if np.isnan(bx[i]) or np.isnan(by[i]):
            poss[i] = cur; continue
        hxs = m["home_x"][i]; hys = m["home_y"][i]
        axs = m["away_x"][i]; ays = m["away_y"][i]
        if not hxs or not axs:
            poss[i] = cur; continue
        hd = np.min(np.hypot(np.array(hxs) - bx[i], np.array(hys) - by[i]))
        ad = np.min(np.hypot(np.array(axs) - bx[i], np.array(ays) - by[i]))
        closer = 0 if hd <= ad else 1
        if closer != cur:
            consec += 1
            if consec >= hold:
                cur = closer; consec = 0
        else:
            consec = 0
        poss[i] = cur
    return poss

# ============================================================
# 3. TURNOVER DETECTION
# ============================================================

def derive_turnovers(m, poss, prox_m=PROX_M, min_hold=HOLD_FRAMES):
    events = []
    n = m["n"]; bx = m["ball_x"]; by = m["ball_y"]
    cur = -1; team_frames = 0
    
    for i in range(n):
        t = poss[i]
        if t == -1: team_frames = 0; continue
        if t != cur:
            if team_frames >= min_hold and cur != -1:
                b_x = bx[i]; b_y = by[i]
                if not np.isnan(b_x):
                    # Defending team is the one that just regained possession
                    def_xs = m["home_x"][i] if t == 0 else m["away_x"][i]
                    def_ys = m["home_y"][i] if t == 0 else m["away_y"][i]
                    if def_xs:
                        dists = np.hypot(np.array(def_xs)-b_x, np.array(def_ys)-b_y)
                        if prox_m >= 999 or np.min(dists) <= prox_m:
                            events.append({"idx": i, "winning_team": t, "min_dist": float(np.min(dists))})
            cur = t; team_frames = 1
        else:
            team_frames += 1
    return events

def label_target(n, turnovers, horizon=HORIZON):
    tgt = np.zeros(n, dtype=np.int8)
    for ev in turnovers:
        idx = ev["idx"]
        start = max(0, idx - horizon)
        tgt[start:idx] = 1
    return tgt

# ============================================================
# 4. VECTORIZED FEATURE EXTRACTION
# ============================================================

FEAT_NAMES = [
    "feat_def_dist_1", "feat_def_dist_2", "feat_def_dist_3", "feat_def_dist_mean3",
    "feat_closing_speed_max", "feat_closing_speed_mean3",
    "feat_pressure_index",
    "feat_def_density_5m", "feat_def_density_10m", "feat_att_density_10m",
    "feat_numerical_advantage_10m",
    "feat_def_hull_area", "feat_def_block_width", "feat_def_block_depth",
    "feat_passing_lane_occlusion",
    "feat_def_dispersion", "feat_def_centroid_dist_to_ball", "feat_def_line_height",
    "feat_ball_speed", "feat_ball_progression_x",
    "feat_roll_pressure_index_mean", "feat_roll_pressure_index_max",
    "feat_roll_closing_speed_mean", "feat_roll_dist_mean3_min", "feat_roll_hull_area_change",
    "feat_ball_x", "feat_ball_y",
    "feat_dist_to_goal", "feat_dist_to_sideline",
    "feat_viable_passing_options", "feat_possession_duration_s",
    "feat_temp_accel", "feat_temp_hull_shrink",
]

def extract_features(m, poss):
    n = m["n"]
    W = 25
    
    # Forward-fill ball coordinates
    bx = pd.Series(m["ball_x"]).ffill().fillna(52.5).values
    by_ = pd.Series(m["ball_y"]).ffill().fillna(34.0).values
    
    # Per-frame spatial features (Python loop — unavoidable for irregular lists)
    d1 = np.full(n, 99.0); d2 = np.full(n, 99.0); d3 = np.full(n, 99.0)
    d_m3 = np.full(n, 99.0)
    pressure = np.zeros(n); def_5m = np.zeros(n); def_10m = np.zeros(n)
    att_10m = np.zeros(n); hull_area = np.zeros(n)
    block_w = np.zeros(n); block_d = np.zeros(n)
    centroid_dist = np.zeros(n); dispersion = np.zeros(n); def_line_h = np.zeros(n)
    viable = np.zeros(n)
    
    for i in range(n):
        bxi = bx[i]; byi = by_[i]
        t = poss[i]
        # Attacker = team in possession; defender = other
        att_t = t if t != -1 else 0
        
        hxs = np.array(m["home_x"][i]); hys = np.array(m["home_y"][i])
        axs = np.array(m["away_x"][i]); ays = np.array(m["away_y"][i])
        
        d_xs = hxs if att_t == 1 else axs  # Defender's x
        d_ys = hys if att_t == 1 else ays
        a_xs = axs if att_t == 1 else hxs  # Attacker's x
        a_ys = ays if att_t == 1 else hys
        
        if len(d_xs) > 0:
            dists = np.sort(np.hypot(d_xs - bxi, d_ys - byi))
            d1[i] = dists[0] if len(dists) > 0 else 99
            d2[i] = dists[1] if len(dists) > 1 else 99
            d3[i] = dists[2] if len(dists) > 2 else 99
            d_m3[i] = np.mean(dists[:3]) if len(dists) >= 3 else np.mean(dists)
            pressure[i] = max(0.0, 1.0 - d1[i] / 10.0) if d1[i] < 10 else 0.0
            def_5m[i] = np.sum(dists <= 5.0)
            def_10m[i] = np.sum(dists <= 10.0)
            dispersion[i] = np.std(dists) if len(dists) > 1 else 0.0
            def_line_h[i] = np.mean(d_xs)
            if len(d_xs) >= 3:
                try:
                    pts = np.column_stack([d_xs, d_ys])
                    hull_area[i] = ConvexHull(pts).volume
                except: pass
            block_w[i] = d_xs.max() - d_xs.min() if len(d_xs) > 1 else 0
            block_d[i] = d_ys.max() - d_ys.min() if len(d_ys) > 1 else 0
            cxm = np.mean(d_xs); cym = np.mean(d_ys)
            centroid_dist[i] = math.hypot(cxm - bxi, cym - byi)
        
        if len(a_xs) > 0:
            a_dists = np.hypot(a_xs - bxi, a_ys - byi)
            att_10m[i] = np.sum(a_dists <= 10.0)
            viable[i] = max(0, int(att_10m[i]) - int(def_10m[i]))
    
    # Scalar features (vectorized)
    dist_to_goal = np.hypot(bx - 105.0, by_ - 34.0)
    dist_to_sideline = np.minimum(by_, 68.0 - by_)
    
    # Ball kinematics
    ball_speed = np.abs(np.diff(bx, prepend=bx[0])) * HZ
    ball_prog = pd.Series(bx).diff(W).fillna(0).values
    
    # Closing speed from diff of d_m3
    closing_max = pd.Series(d_m3).diff(1).abs().fillna(0).values
    closing_m3 = pd.Series(d_m3).diff(3).abs().rolling(3).mean().fillna(0).values
    
    # Rolling features
    s_pressure = pd.Series(pressure)
    roll_p_mean = s_pressure.rolling(W, min_periods=1).mean().values
    roll_p_max = s_pressure.rolling(W, min_periods=1).max().values
    roll_cs_mean = pd.Series(closing_max).rolling(W, min_periods=1).mean().values
    roll_dist_min = pd.Series(d_m3).rolling(W, min_periods=1).min().values
    roll_hull_chg = pd.Series(hull_area).diff(W).fillna(0).values
    
    # Temporal derivatives
    temp_accel = pd.Series(closing_max).diff(W).fillna(0).values
    temp_hull_shrink = pd.Series(hull_area).diff(W).fillna(0).values
    
    X = np.column_stack([
        d1, d2, d3, d_m3,
        closing_max, closing_m3,
        pressure, def_5m, def_10m, att_10m,
        def_10m - att_10m,
        hull_area, block_w, block_d,
        np.zeros(n),  # passing_lane_occlusion — unavailable
        dispersion, centroid_dist, def_line_h,
        ball_speed, ball_prog,
        roll_p_mean, roll_p_max, roll_cs_mean, roll_dist_min, roll_hull_chg,
        bx, by_,
        dist_to_goal, dist_to_sideline, viable,
        np.zeros(n),  # possession_duration_s — approximated as 0
        temp_accel, temp_hull_shrink,
    ])
    return X

# ============================================================
# 5. EVALUATION HELPERS
# ============================================================

def eval_frame(y, p, t):
    pred = (p >= t).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"f_prec": prec, "f_rec": rec, "f_f1": f1}

def eval_roc_pr(y, p):
    if y.sum() == 0: return {"roc": 0.5, "pr": 0.0}
    roc = roc_auc_score(y, p)
    pr_c, rc_c, _ = precision_recall_curve(y, p)
    pr = auc(rc_c, pr_c)
    return {"roc": float(roc), "pr": float(pr)}

print("=" * 60)
print("REAL-TO-REAL EXPERIMENT")
print("=" * 60)

# ============================================================
# 6. LOAD ALL THREE MATCHES
# ============================================================

print("\nStep 1: Loading all Metrica matches...")
m = {}
for mid in [1, 2, 3]:
    m[mid] = load_match(mid)

# ============================================================
# 7. INVENTORY
# ============================================================

inv = {}
for mid in [1, 2, 3]:
    bx_ = m[mid]["ball_x"]
    ball_cov = float(np.mean(~np.isnan(bx_)))
    # Ball runs
    runs = []; cur = 0
    for v in ~np.isnan(bx_):
        if v: cur += 1
        else:
            if cur > 0: runs.append(cur); cur = 0
    if cur > 0: runs.append(cur)
    runs = np.array(runs) if runs else np.array([0])
    inv[mid] = {
        "n_frames": m[mid]["n"],
        "fps": HZ,
        "duration_min": round(m[mid]["n"] / HZ / 60, 1),
        "ball_coverage_pct": round(ball_cov * 100, 1),
        "ball_median_run": float(np.median(runs)),
        "ball_max_run": int(np.max(runs)),
        "n_ball_runs": int(len(runs)),
    }
    print(f"  Match {mid}: {inv[mid]['n_frames']} frames | {inv[mid]['duration_min']} min | ball {inv[mid]['ball_coverage_pct']}%")

with open(f"{OUT}/REAL_DATA_INVENTORY.json", "w") as f:
    json.dump(inv, f, indent=4)

# ============================================================
# 8. POSSESSION & TURNOVER FOR EACH MATCH
# ============================================================

print("\nStep 2: Possession & turnovers...")
poss = {}; turnovers_3m = {}; turnovers_unc = {}; y_all = {}

for mid in [1, 2, 3]:
    print(f"  Match {mid} possession...")
    poss[mid] = assign_possession(m[mid])
    turnovers_3m[mid] = derive_turnovers(m[mid], poss[mid], prox_m=3.0)
    turnovers_unc[mid] = derive_turnovers(m[mid], poss[mid], prox_m=999.0)
    y_all[mid] = label_target(m[mid]["n"], turnovers_3m[mid])
    print(f"    Events 3m: {len(turnovers_3m[mid])}, Uncoupled: {len(turnovers_unc[mid])}, Positive: {y_all[mid].sum()}")

target_info = {mid: {
    "n_events_3m": len(turnovers_3m[mid]),
    "n_events_uncoupled": len(turnovers_unc[mid]),
    "positive_frames": int(y_all[mid].sum()),
    "prevalence": float(y_all[mid].mean()),
} for mid in [1,2,3]}

with open(f"{OUT}/REAL_TARGET_CONSTRUCTION.json", "w") as f:
    json.dump(target_info, f, indent=4)

# ============================================================
# 9. FEATURE EXTRACTION
# ============================================================

print("\nStep 3: Feature extraction (slow — ~5 min per match)...")
X_all = {}
for mid in [1, 2, 3]:
    t0 = time.time()
    print(f"  Match {mid} features...")
    X_all[mid] = extract_features(m[mid], poss[mid])
    print(f"    Done in {time.time()-t0:.1f}s, shape={X_all[mid].shape}")

# ============================================================
# 10. EXPERIMENTAL DESIGN
# ============================================================

# TRAIN:  Match 2 (full)
# VAL:    Match 3, first 70% (minus purge)
# TEST:   Match 3, final 30% (NEVER TOUCHED until final eval)

n3 = m[3]["n"]
val_split = int(n3 * 0.70)

X_train = X_all[2]
y_train = y_all[2]

X_val   = X_all[3][:val_split - PURGE]
y_val   = y_all[3][:val_split - PURGE]

X_test  = X_all[3][val_split:]
y_test  = y_all[3][val_split:]

print(f"\nDesign: TRAIN=Match2({len(y_train)}fr), VAL=Match3 first70%({len(y_val)}fr), TEST=Match3 last30%({len(y_test)}fr)")
print(f"Train positives: {y_train.sum()}, Val positives: {y_val.sum()}, Test positives: {y_test.sum()}")

# ============================================================
# 11. EXPERIMENT C — SMOOTHING SENSITIVITY (Validation only)
# ============================================================

print("\nStep 4: Smoothing sensitivity on validation...")
clf_base = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf_base.fit(X_train, y_train)

p_val_raw = clf_base.predict_proba(X_val)[:, 1]

smooth_results = {}
best_smooth_f1 = -1; best_smooth_w = 0
for w_sec in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    w_frames = max(1, int(w_sec * HZ))
    if w_sec == 0:
        p_s = p_val_raw
    else:
        p_s = pd.Series(p_val_raw).rolling(w_frames, min_periods=1, center=False).mean().values
    
    # Threshold sweep on val to get best F1 per smoothing window
    best_f1_w = 0; best_t_w = 0.10
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        ev = evaluate_events_strict(y_val, p_s, t)
        if ev["f1"] > best_f1_w:
            best_f1_w = ev["f1"]; best_t_w = t
    
    smooth_results[f"{w_sec}s"] = {"best_val_f1": best_f1_w, "best_threshold": best_t_w}
    if best_f1_w > best_smooth_f1:
        best_smooth_f1 = best_f1_w; best_smooth_w = w_sec; best_smooth_t = best_t_w
    print(f"  Smooth={w_sec}s: best_val_F1={best_f1_w:.3f} @ t={best_t_w}")

print(f"\n  FROZEN: smoothing={best_smooth_w}s, threshold={best_smooth_t}")
smooth_results["_frozen"] = {"smooth_window_s": best_smooth_w, "threshold": best_smooth_t, "val_f1": best_smooth_f1}

with open(f"{OUT}/REAL_SMOOTHING_RESULTS.json", "w") as f:
    json.dump(smooth_results, f, indent=4)

# ============================================================
# 12. EXPERIMENT D — MODEL COMPARISON (Validation only)
# ============================================================

print("\nStep 5: Model comparison on validation...")

def eval_model_val(clf_m, w_s, t):
    clf_m.fit(X_train, y_train)
    p_v = clf_m.predict_proba(X_val)[:, 1]
    if w_s > 0:
        p_v = pd.Series(p_v).rolling(int(w_s*HZ), min_periods=1, center=False).mean().values
    ev = evaluate_events_strict(y_val, p_v, t)
    rp = eval_roc_pr(y_val, p_v)
    ev.pop("pred_events", None); ev.pop("true_events", None)
    return {"event_f1": ev["f1"], "roc_auc": rp["roc"], "pr_auc": rp["pr"], **ev}

models = {
    "LogReg": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "HistGB": HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42),
    "HistGB_balanced": HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42, class_weight="balanced"),
}

model_results = {}
for name, mdl in models.items():
    r = eval_model_val(mdl, best_smooth_w, best_smooth_t)
    model_results[name] = r
    print(f"  {name}: F1={r['event_f1']:.3f} ROC={r['roc_auc']:.3f}")

with open(f"{OUT}/REAL_MODEL_COMPARISON.json", "w") as f:
    json.dump(model_results, f, indent=4)

# ============================================================
# 13. FINAL FROZEN MODEL — EVALUATE ON TEST (once)
# ============================================================

print("\nStep 6: FINAL TEST EVALUATION (Match 3, last 30%)...")

# Apply frozen configuration
w_frames_final = max(1, int(best_smooth_w * HZ))
clf_final = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf_final.fit(X_train, y_train)

p_test_raw = clf_final.predict_proba(X_test)[:, 1]
if best_smooth_w > 0:
    p_test_s = pd.Series(p_test_raw).rolling(w_frames_final, min_periods=1, center=False).mean().values
else:
    p_test_s = p_test_raw

ev_test = evaluate_events_strict(y_test, p_test_s, best_smooth_t)
fm_test = eval_frame(y_test, p_test_s, best_smooth_t)
rp_test = eval_roc_pr(y_test, p_test_s)

# Also report at threshold 0.50 for comparability with synthetic experiment
ev_test_050 = evaluate_events_strict(y_test, p_test_s, 0.50)

test_results = {
    "dataset": "Metrica Match 3 — final 30% (chronological holdout)",
    "frozen_config": {"smooth_window_s": best_smooth_w, "threshold": best_smooth_t},
    "at_frozen_threshold": {
        "event_f1": ev_test["f1"],
        "event_prec": ev_test["prec"],
        "event_rec": ev_test["rec"],
        "tp": ev_test["tp"], "fp": ev_test["fp"], "fn": ev_test["fn"],
        "n_gt": ev_test["n_gt"], "n_pred": ev_test["n_pred"],
        **fm_test, **rp_test,
    },
    "at_050_for_comparability": {
        "event_f1": ev_test_050["f1"],
        "tp": ev_test_050["tp"], "fp": ev_test_050["fp"], "fn": ev_test_050["fn"],
    },
}
for d in [ev_test, ev_test_050]:
    d.pop("pred_events", None); d.pop("true_events", None)

with open(f"{OUT}/REAL_BASELINE_RESULTS.json", "w") as f:
    json.dump(test_results, f, indent=4)

print(f"\n=== REAL->REAL TEST RESULTS ===")
print(f"Frozen: smooth={best_smooth_w}s, threshold={best_smooth_t}")
print(f"Event F1: {ev_test['f1']:.3f}, Precision: {ev_test['prec']:.3f}, Recall: {ev_test['rec']:.3f}")
print(f"TP: {ev_test['tp']}, FP: {ev_test['fp']}, FN: {ev_test['fn']}")
print(f"GT Events: {ev_test['n_gt']}, Pred Events: {ev_test['n_pred']}")
print(f"ROC-AUC: {rp_test['roc']:.3f}, PR-AUC: {rp_test['pr']:.3f}")

# ============================================================
# 14. BOOTSTRAP UNCERTAINTY
# ============================================================

tp_i = ev_test["tp"]; fp_i = ev_test["fp"]; fn_i = ev_test["fn"]
n_gt = ev_test["n_gt"]

# Re-run strict evaluator to get matched event lists
ev_full = evaluate_events_strict(y_test, p_test_s, best_smooth_t)
pred_ev = ev_full.get("pred_events", [])
true_ev = ev_full.get("true_events", [])

# Identify TPs
t_matched = set()
for p_s2, p_e2 in pred_ev:
    for t_idx, (t_s2, t_e2) in enumerate(true_ev):
        t_exp_s = max(0, t_s2 - 50); t_exp_e = t_e2 + 50
        if p_s2 <= t_exp_e and p_e2 >= t_exp_s:
            t_matched.add(t_idx)

np.random.seed(42)
f1s = []
t_inds = np.arange(len(true_ev))
for _ in range(2000):
    samp = np.random.choice(t_inds, size=len(true_ev), replace=True)
    s_tp = sum(1 for idx in samp if idx in t_matched)
    s_fn = len(samp) - s_tp; s_fp = fp_i
    s_prec = s_tp / (s_tp + s_fp) if s_tp + s_fp > 0 else 0
    s_rec = s_tp / (s_tp + s_fn) if s_tp + s_fn > 0 else 0
    f1s.append(2 * s_prec * s_rec / (s_prec + s_rec) if s_prec + s_rec > 0 else 0)

f1s = np.array(f1s)
ci_lo, ci_hi = np.percentile(f1s, [2.5, 97.5])
uncert = {"f1_point": ev_test["f1"], "f1_mean_boot": float(np.mean(f1s)),
          "ci_2.5": float(ci_lo), "ci_97.5": float(ci_hi), "n_gt_events": n_gt}

with open(f"{OUT}/REAL_UNCERTAINTY.json", "w") as f:
    json.dump(uncert, f, indent=4)

print(f"\nUncertainty: F1={ev_test['f1']:.3f}, 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}] (n={n_gt} events)")

# ============================================================
# 15. SYNTHETIC vs REAL COMPARISON SUMMARY
# ============================================================

comparison = {
    "synthetic_to_synthetic": {
        "train": "Synthetic Match 1 (70%)",
        "test": "Synthetic Match 1 (30%)",
        "event_f1": 0.600, "roc_auc": 0.927, "pr_auc": 0.563,
        "tp": 9, "fp": 2, "fn": 10, "n_gt": 19,
        "threshold": 0.50, "smooth_window_s": 1.0,
    },
    "synthetic_to_real": {
        "train": "Synthetic Match 1 (70%)",
        "test": "Metrica Match 2 (real, full)",
        "event_f1": 0.000, "roc_auc": 0.755, "pr_auc": 0.034,
        "tp": 0, "fp": 0, "fn": 20, "n_gt": 20,
        "threshold": 0.50, "note": "Frozen threshold — never triggered on real data",
    },
    "real_to_real": {
        "train": "Metrica Match 2 (real, full)",
        "test": "Metrica Match 3 (real, final 30%)",
        "event_f1": ev_test["f1"],
        "roc_auc": rp_test["roc"], "pr_auc": rp_test["pr"],
        "tp": ev_test["tp"], "fp": ev_test["fp"], "fn": ev_test["fn"],
        "n_gt": ev_test["n_gt"],
        "threshold": best_smooth_t, "smooth_window_s": best_smooth_w,
        "f1_95ci": [float(ci_lo), float(ci_hi)],
    },
}

with open(f"{OUT}/SYNTHETIC_VS_REAL_COMPARISON.json", "w") as f:
    json.dump(comparison, f, indent=4)

print("\nAll results saved to", OUT)
print("Done.")
