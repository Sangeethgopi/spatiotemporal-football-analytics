"""
Recovery script: re-runs only the TEST evaluation with frozen config.
Avoids re-doing feature extraction.
Frozen config (selected on validation only):
  - smoothing = 3.0s (75 frames)
  - threshold = 0.15
  - model = HistGradientBoosting (same hyperparams)
  - train = Match 2, val+test = Match 3
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from kloppy import metrica
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.abspath('.'))
from scripts.run_apples_apples import evaluate_events_strict

PITCH_L = 105.0; PITCH_W = 68.0; HZ = 25
HOLD_FRAMES = 75; HORIZON = 100; PROX_M = 3.0; PURGE = 100
OUT = "data/results/real_to_real"
os.makedirs(OUT, exist_ok=True)

# FROZEN CONFIG (selected on validation)
BEST_SMOOTH_W = 3.0  # seconds
BEST_SMOOTH_FRAMES = 75
BEST_THRESH = 0.15

def load_match(match_id):
    t0 = time.time()
    print(f"  Loading Match {match_id}...")
    ds = metrica.load_open_data(match_id=match_id, sample_rate=None, limit=None)
    ball_x = []; ball_y = []
    home_x = []; home_y = []; away_x = []; away_y = []
    for frame in ds.frames:
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
    n = len(ball_x)
    print(f"    {n} frames in {time.time()-t0:.1f}s")
    return {"n": n, "ball_x": np.array(ball_x), "ball_y": np.array(ball_y),
            "home_x": home_x, "home_y": home_y, "away_x": away_x, "away_y": away_y}

def assign_possession(m, hold=HOLD_FRAMES):
    n = m["n"]; bx = m["ball_x"]; by_ = m["ball_y"]
    poss = np.full(n, -1, dtype=int); cur = -1; consec = 0
    for i in range(n):
        if np.isnan(bx[i]) or np.isnan(by_[i]): poss[i] = cur; continue
        hxs = m["home_x"][i]; hys = m["home_y"][i]
        axs = m["away_x"][i]; ays = m["away_y"][i]
        if not hxs or not axs: poss[i] = cur; continue
        hd = np.min(np.hypot(np.array(hxs)-bx[i], np.array(hys)-by_[i]))
        ad = np.min(np.hypot(np.array(axs)-bx[i], np.array(ays)-by_[i]))
        closer = 0 if hd <= ad else 1
        if closer != cur:
            consec += 1
            if consec >= hold: cur = closer; consec = 0
        else: consec = 0
        poss[i] = cur
    return poss

def derive_turnovers(m, poss, prox_m=PROX_M, min_hold=HOLD_FRAMES):
    events = []; n = m["n"]; bx = m["ball_x"]; by_ = m["ball_y"]
    cur = -1; tf = 0
    for i in range(n):
        t = poss[i]
        if t == -1: tf = 0; continue
        if t != cur:
            if tf >= min_hold and cur != -1:
                bxi = bx[i]; byi = by_[i]
                if not np.isnan(bxi):
                    dxs = m["home_x"][i] if t == 0 else m["away_x"][i]
                    dys = m["home_y"][i] if t == 0 else m["away_y"][i]
                    if dxs:
                        dists = np.hypot(np.array(dxs)-bxi, np.array(dys)-byi)
                        if prox_m >= 999 or np.min(dists) <= prox_m:
                            events.append({"idx": i, "winning_team": t, "min_dist": float(np.min(dists))})
            cur = t; tf = 1
        else: tf += 1
    return events

def label_target(n, turnovers, horizon=HORIZON):
    tgt = np.zeros(n, dtype=np.int8)
    for ev in turnovers:
        idx = ev["idx"]
        start = max(0, idx - horizon)
        tgt[start:idx] = 1
    return tgt

def extract_features(m, poss):
    n = m["n"]; W = 25
    bx = pd.Series(m["ball_x"]).ffill().fillna(52.5).values
    by_ = pd.Series(m["ball_y"]).ffill().fillna(34.0).values
    d1 = np.full(n, 99.0); d2 = np.full(n, 99.0); d3 = np.full(n, 99.0)
    d_m3 = np.full(n, 99.0); pressure = np.zeros(n)
    def_5m = np.zeros(n); def_10m = np.zeros(n); att_10m = np.zeros(n)
    hull_area = np.zeros(n); block_w = np.zeros(n); block_d = np.zeros(n)
    centroid_dist = np.zeros(n); dispersion = np.zeros(n); def_line_h = np.zeros(n)
    viable = np.zeros(n)
    for i in range(n):
        bxi = bx[i]; byi = by_[i]
        t = poss[i]; att_t = t if t != -1 else 0
        hxs = np.array(m["home_x"][i]); hys = np.array(m["home_y"][i])
        axs = np.array(m["away_x"][i]); ays = np.array(m["away_y"][i])
        dxs = hxs if att_t == 1 else axs; dys = hys if att_t == 1 else ays
        axs2 = axs if att_t == 1 else hxs; ays2 = ays if att_t == 1 else hys
        if len(dxs) > 0:
            dists = np.sort(np.hypot(dxs-bxi, dys-byi))
            d1[i] = dists[0] if len(dists)>0 else 99
            d2[i] = dists[1] if len(dists)>1 else 99
            d3[i] = dists[2] if len(dists)>2 else 99
            d_m3[i] = np.mean(dists[:3]) if len(dists)>=3 else np.mean(dists)
            pressure[i] = max(0.0, 1.0-d1[i]/10.0) if d1[i]<10 else 0.0
            def_5m[i] = np.sum(dists<=5.0); def_10m[i] = np.sum(dists<=10.0)
            dispersion[i] = np.std(dists) if len(dists)>1 else 0.0
            def_line_h[i] = np.mean(dxs)
            if len(dxs)>=3:
                try:
                    pts = np.column_stack([dxs, dys])
                    hull_area[i] = ConvexHull(pts).volume
                except: pass
            block_w[i] = dxs.max()-dxs.min() if len(dxs)>1 else 0
            block_d[i] = dys.max()-dys.min() if len(dys)>1 else 0
            cxm = np.mean(dxs); cym = np.mean(dys)
            centroid_dist[i] = math.hypot(cxm-bxi, cym-byi)
        if len(axs2)>0:
            ad = np.hypot(axs2-bxi, ays2-byi)
            att_10m[i] = np.sum(ad<=10.0)
            viable[i] = max(0, int(att_10m[i])-int(def_10m[i]))
    dist_goal = np.hypot(bx-105.0, by_-34.0)
    dist_side = np.minimum(by_, 68.0-by_)
    ball_spd = np.abs(np.diff(bx, prepend=bx[0]))*HZ
    ball_prog = pd.Series(bx).diff(W).fillna(0).values
    closing_max = pd.Series(d_m3).diff(1).abs().fillna(0).values
    closing_m3 = pd.Series(d_m3).diff(3).abs().rolling(3).mean().fillna(0).values
    s_p = pd.Series(pressure)
    roll_p_mean = s_p.rolling(W, min_periods=1).mean().values
    roll_p_max = s_p.rolling(W, min_periods=1).max().values
    roll_cs_mean = pd.Series(closing_max).rolling(W, min_periods=1).mean().values
    roll_dist_min = pd.Series(d_m3).rolling(W, min_periods=1).min().values
    roll_hull_chg = pd.Series(hull_area).diff(W).fillna(0).values
    temp_accel = pd.Series(closing_max).diff(W).fillna(0).values
    temp_hull_shrink = pd.Series(hull_area).diff(W).fillna(0).values
    return np.column_stack([
        d1, d2, d3, d_m3, closing_max, closing_m3, pressure,
        def_5m, def_10m, att_10m, def_10m-att_10m,
        hull_area, block_w, block_d, np.zeros(n), dispersion, centroid_dist, def_line_h,
        ball_spd, ball_prog, roll_p_mean, roll_p_max, roll_cs_mean, roll_dist_min, roll_hull_chg,
        bx, by_, dist_goal, dist_side, viable, np.zeros(n),
        temp_accel, temp_hull_shrink,
    ])

def eval_roc_pr(y, p):
    if y.sum() == 0: return {"roc": 0.5, "pr": 0.0}
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    roc = roc_auc_score(y, p)
    pr_c, rc_c, _ = precision_recall_curve(y, p)
    pr = auc(rc_c, pr_c)
    return {"roc": float(roc), "pr": float(pr)}

def eval_frame_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(np.sum((pred==1)&(y==1))); fp = int(np.sum((pred==1)&(y==0))); fn = int(np.sum((pred==0)&(y==1)))
    prec = tp/(tp+fp) if tp+fp>0 else 0.0; rec = tp/(tp+fn) if tp+fn>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
    return {"f_prec": prec, "f_rec": rec, "f_f1": f1}

print("Loading Match 2 (TRAIN)...")
m2 = load_match(2)
print("Loading Match 3 (VAL+TEST)...")
m3 = load_match(3)

print("Possession Match 2..."); poss2 = assign_possession(m2)
print("Possession Match 3..."); poss3 = assign_possession(m3)

print("Turnovers Match 2...")
t2_3m = derive_turnovers(m2, poss2, prox_m=3.0)
t2_unc = derive_turnovers(m2, poss2, prox_m=999.0)
y2 = label_target(m2["n"], t2_3m)
print(f"  M2 events 3m: {len(t2_3m)}, uncoupled: {len(t2_unc)}, positives: {y2.sum()}")

print("Turnovers Match 3...")
t3_3m = derive_turnovers(m3, poss3, prox_m=3.0)
t3_unc = derive_turnovers(m3, poss3, prox_m=999.0)
y3 = label_target(m3["n"], t3_3m)
n3 = m3["n"]; val_split = int(n3 * 0.70)
y3_val = y3[:val_split - PURGE]
y3_test = y3[val_split:]
print(f"  M3 events 3m: {len(t3_3m)}, uncoupled: {len(t3_unc)}")
print(f"  Val positives: {y3_val.sum()}, Test positives: {y3_test.sum()}")

print("Feature extraction Match 2 (train)...")
t0 = time.time(); X2 = extract_features(m2, poss2); print(f"  Done {time.time()-t0:.1f}s")

print("Feature extraction Match 3 (val+test)...")
t0 = time.time(); X3 = extract_features(m3, poss3); print(f"  Done {time.time()-t0:.1f}s")

X_train = X2; y_train = y2
X_val = X3[:val_split - PURGE]; X_test = X3[val_split:]

print("\nFitting frozen model...")
clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf.fit(X_train, y_train)

print("Evaluating on TEST set (Match 3, final 30%)...")
p_test_raw = clf.predict_proba(X_test)[:, 1]
p_test_s = pd.Series(p_test_raw).rolling(BEST_SMOOTH_FRAMES, min_periods=1, center=False).mean().values

ev_test = evaluate_events_strict(y3_test, p_test_s, BEST_THRESH)
ev_test_050 = evaluate_events_strict(y3_test, p_test_s, 0.50)
fm_test = eval_frame_metrics(y3_test, p_test_s, BEST_THRESH)
rp_test = eval_roc_pr(y3_test, p_test_s)

# Also evaluate at threshold 0.05 (diagnostic)
ev_005 = evaluate_events_strict(y3_test, p_test_s, 0.05)

print("\n=== REAL->REAL TEST RESULTS ===")
print(f"Frozen config: smooth={BEST_SMOOTH_W}s ({BEST_SMOOTH_FRAMES}fr), threshold={BEST_THRESH}")
print(f"Event F1: {ev_test['f1']:.3f}  Precision: {ev_test['prec']:.3f}  Recall: {ev_test['rec']:.3f}")
print(f"TP: {ev_test['tp']}, FP: {ev_test['fp']}, FN: {ev_test['fn']}")
print(f"GT Events: {ev_test['n_gt']}, Pred Events: {ev_test['n_pred']}")
print(f"ROC-AUC: {rp_test['roc']:.3f}, PR-AUC: {rp_test['pr']:.3f}")
print(f"At t=0.50: F1={ev_test_050['f1']:.3f} TP={ev_test_050['tp']} FP={ev_test_050['fp']} FN={ev_test_050['fn']}")

# Bootstrap uncertainty
pred_ev_full = ev_test.get("pred_events", [])
true_ev_full = ev_test.get("true_events", [])
t_matched = set()
for ps2, pe2 in pred_ev_full:
    for tidx, (ts2, te2) in enumerate(true_ev_full):
        if ps2 <= te2+50 and pe2 >= max(0, ts2-50):
            t_matched.add(tidx)
np.random.seed(42)
fp_orig = ev_test["fp"]
f1s = []
t_inds = np.arange(len(true_ev_full))
for _ in range(2000):
    samp = np.random.choice(t_inds, size=len(true_ev_full), replace=True)
    stp = sum(1 for idx in samp if idx in t_matched)
    sfn = len(samp)-stp; sfp = fp_orig
    sp = stp/(stp+sfp) if stp+sfp>0 else 0
    sr = stp/(stp+sfn) if stp+sfn>0 else 0
    f1s.append(2*sp*sr/(sp+sr) if sp+sr>0 else 0)
f1s = np.array(f1s)
ci_lo, ci_hi = np.percentile(f1s, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] (n={ev_test['n_gt']} GT events)")

for d in [ev_test, ev_test_050, ev_005]:
    d.pop("pred_events", None); d.pop("true_events", None)

# Save all results
results = {
    "design": {
        "train": "Metrica Match 2 (real, full, 141156 frames)",
        "val": "Metrica Match 3, first 70% minus purge (100532 frames)",
        "test": "Metrica Match 3, final 30% (43129 frames)",
        "note": "TEST was untouched until this final evaluation"
    },
    "frozen_config": {"smooth_window_s": BEST_SMOOTH_W, "smooth_frames": BEST_SMOOTH_FRAMES,
                      "threshold": BEST_THRESH, "model": "HistGradientBoosting"},
    "validation_results": {
        "smoothing_sensitivity": {
            "0s":   {"val_f1": 0.366, "threshold": 0.50},
            "0.5s": {"val_f1": 0.496, "threshold": 0.40},
            "1.0s": {"val_f1": 0.515, "threshold": 0.25},
            "1.5s": {"val_f1": 0.562, "threshold": 0.25},
            "2.0s": {"val_f1": 0.591, "threshold": 0.20},
            "3.0s": {"val_f1": 0.618, "threshold": 0.15},
        },
        "model_comparison": {
            "LogReg":          {"val_f1": 0.502, "roc_auc": 0.801},
            "RandomForest":    {"val_f1": 0.461, "roc_auc": 0.827},
            "HistGB":          {"val_f1": 0.618, "roc_auc": 0.825},
            "HistGB_balanced": {"val_f1": 0.389, "roc_auc": 0.826},
        }
    },
    "test_results": {
        "at_frozen_threshold": {
            "event_f1": ev_test["f1"], "event_prec": ev_test["prec"], "event_rec": ev_test["rec"],
            "tp": ev_test["tp"], "fp": ev_test["fp"], "fn": ev_test["fn"],
            "n_gt": ev_test["n_gt"], "n_pred": ev_test["n_pred"],
            "roc_auc": rp_test["roc"], "pr_auc": rp_test["pr"],
            **fm_test,
        },
        "at_050_for_comparability": {
            "event_f1": ev_test_050["f1"],
            "tp": ev_test_050["tp"], "fp": ev_test_050["fp"], "fn": ev_test_050["fn"],
        },
        "bootstrap_uncertainty": {
            "f1_point": ev_test["f1"],
            "f1_mean_boot": float(np.mean(f1s)),
            "ci_2.5": float(ci_lo), "ci_97.5": float(ci_hi),
            "n_gt_events": ev_test["n_gt"],
        }
    },
    "comparison_with_synthetic": {
        "synthetic_to_synthetic": {"event_f1": 0.600, "threshold": 0.50, "smooth_s": 1.0,
                                   "n_gt": 19, "roc_auc": 0.927},
        "synthetic_to_real":      {"event_f1": 0.000, "threshold": 0.50, "smooth_s": 1.0,
                                   "n_gt": 20, "roc_auc": 0.755},
        "real_to_real":           {"event_f1": ev_test["f1"], "threshold": BEST_THRESH,
                                   "smooth_s": BEST_SMOOTH_W, "n_gt": ev_test["n_gt"],
                                   "roc_auc": rp_test["roc"],
                                   "f1_ci": [float(ci_lo), float(ci_hi)]},
    }
}

with open(f"{OUT}/REAL_BASELINE_RESULTS.json", "w") as f:
    json.dump(results, f, indent=4)

with open(f"{OUT}/REAL_UNCERTAINTY.json", "w") as f:
    json.dump(results["test_results"]["bootstrap_uncertainty"], f, indent=4)

with open(f"{OUT}/SYNTHETIC_VS_REAL_COMPARISON.json", "w") as f:
    json.dump(results["comparison_with_synthetic"], f, indent=4)

with open(f"{OUT}/REAL_DATA_INVENTORY.json", "w") as f:
    json.dump({
        "match_2": {"n_frames": m2["n"], "fps": 25, "ball_coverage": 59.0,
                    "events_3m": len(t2_3m), "events_uncoupled": len(t2_unc)},
        "match_3": {"n_frames": m3["n"], "fps": 25, "ball_coverage": 67.8,
                    "events_3m": len(t3_3m), "events_uncoupled": len(t3_unc)},
    }, f, indent=4)

print("\nAll results saved to", OUT)
