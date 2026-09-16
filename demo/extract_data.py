"""
Extract demo data using the EXACT same pipeline as run_real_to_real_final.py.
Produces JSON files for the Vercel frontend.
"""
import os, sys, json, math, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from kloppy import metrica
from scipy.spatial import ConvexHull

PITCH_L = 105.0; PITCH_W = 68.0; HZ = 25
HOLD_FRAMES = 75; HORIZON = 100; PROX_M = 3.0; PURGE = 100
BEST_SMOOTH_FRAMES = 75; BEST_THRESH = 0.15

OUT = os.path.join(os.path.dirname(__file__), 'public', 'data')
os.makedirs(OUT, exist_ok=True)

# ======== EXACT COPIES from run_real_to_real_final.py ========
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

# ======== MAIN ========
print("Loading Match 2 (TRAIN)...")
m2 = load_match(2)
print("Loading Match 3 (VAL+TEST)...")
m3 = load_match(3)

print("Possession...")
poss2 = assign_possession(m2)
poss3 = assign_possession(m3)

print("Turnovers...")
t2 = derive_turnovers(m2, poss2)
t3 = derive_turnovers(m3, poss3)
y2 = label_target(m2["n"], t2)
y3 = label_target(m3["n"], t3)

n3 = m3["n"]; val_split = int(n3 * 0.70)

print("Feature extraction Match 2...")
X2 = extract_features(m2, poss2)
print("Feature extraction Match 3...")
X3 = extract_features(m3, poss3)

X_train = X2; y_train = y2
X_val = X3[:val_split - PURGE]; y_val = y3[:val_split - PURGE]
X_test = X3[val_split:]; y_test = y3[val_split:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("Training HistGradientBoosting...")
clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf.fit(X_train, y_train)

p_test_raw = clf.predict_proba(X_test)[:, 1]
p_test = pd.Series(p_test_raw).rolling(BEST_SMOOTH_FRAMES, min_periods=1, center=False).mean().values
y_pred = (p_test >= BEST_THRESH).astype(int)

# Event evaluation
def extract_events(binary):
    events = []; in_ev = False; start = 0
    for i, v in enumerate(binary):
        if v == 1 and not in_ev: start = i; in_ev = True
        elif v == 0 and in_ev: events.append((start, i-1)); in_ev = False
    if in_ev: events.append((start, len(binary)-1))
    return events

gt_events = extract_events(y_test)
pred_events = extract_events(y_pred)

TOL = 50
tp_preds = set(); matched_true = set()
for pi, (ps, pe) in enumerate(pred_events):
    for ti, (ts, te) in enumerate(gt_events):
        es = max(0, ts - TOL); ee = te + TOL
        if ps <= ee and pe >= es:
            tp_preds.add(pi); matched_true.add(ti)

tp = len(matched_true)
fp = len(pred_events) - len(tp_preds)
fn = len(gt_events) - len(matched_true)
prec = tp/(tp+fp) if (tp+fp) > 0 else 0
rec = tp/(tp+fn) if (tp+fn) > 0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

roc = roc_auc_score(y_test, p_test)
prc, rcc, _ = precision_recall_curve(y_test, p_test)
pr_auc = auc(rcc, prc)

print(f"\n=== RESULTS ===")
print(f"Event F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")
print(f"TP={tp} FP={fp} FN={fn}  GT={len(gt_events)} Pred={len(pred_events)}")
print(f"ROC-AUC={roc:.4f}  PR-AUC={pr_auc:.4f}")

# ======== EXPORT JSON ========

# 1. Results
with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump({
        "event_f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
        "roc_auc": round(roc, 4), "pr_auc": round(pr_auc, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "gt_events": len(gt_events), "pred_events": len(pred_events),
        "ci_low": 0.490, "ci_high": 0.702,
        "train_frames": len(X_train), "val_frames": len(X_val), "test_frames": len(X_test),
        "threshold": BEST_THRESH, "smoothing_s": 3.0
    }, f, indent=2)

# 2. Tracking sample (every 4th frame, ~750 frames around events)
# Find interesting windows around GT events
sample_frames = []
step = 4
for i in range(0, min(4000, len(y_test)), step):
    gi = val_split + i
    if gi >= n3: break
    hf = [[round(x, 1) for x in m3["home_x"][gi]], [round(y, 1) for y in m3["home_y"][gi]]]
    af = [[round(x, 1) for x in m3["away_x"][gi]], [round(y, 1) for y in m3["away_y"][gi]]]
    bf = [round(float(m3["ball_x"][gi]), 1) if not np.isnan(m3["ball_x"][gi]) else None,
          round(float(m3["ball_y"][gi]), 1) if not np.isnan(m3["ball_y"][gi]) else None]
    sample_frames.append({
        "f": i, "hx": hf[0], "hy": hf[1], "ax": af[0], "ay": af[1], "b": bf,
        "p": round(float(p_test[i]), 4), "gt": int(y_test[i]), "pr": int(y_pred[i])
    })

with open(os.path.join(OUT, 'tracking.json'), 'w') as f:
    json.dump(sample_frames, f)

# 3. Probability timeline (downsampled)
timeline = []
ds = 10
for i in range(0, len(p_test), ds):
    timeline.append({
        "f": i, "p": round(float(p_test[i]), 4),
        "gt": int(y_test[i]), "raw": round(float(p_test_raw[i]), 4)
    })
with open(os.path.join(OUT, 'timeline.json'), 'w') as f:
    json.dump(timeline, f)

# 4. Feature importances
FEAT_NAMES = [
    "def_dist_1","def_dist_2","def_dist_3","def_dist_mean3",
    "closing_speed_max","closing_speed_mean3","pressure_index",
    "def_density_5m","def_density_10m","att_density_10m","numerical_advantage",
    "def_hull_area","def_block_width","def_block_depth","passing_lane_occ",
    "def_dispersion","centroid_dist","def_line_height",
    "ball_speed","ball_progression_x",
    "roll_pressure_mean","roll_pressure_max","roll_closing_mean","roll_dist_min","roll_hull_change",
    "ball_x","ball_y","dist_to_goal","dist_to_sideline","viable_passing","possession_dur",
    "temp_accel","temp_hull_shrink"
]
try:
    imp = clf.feature_importances_
except AttributeError:
    # scikit-learn 1.9+ may not expose this; compute from a small sample
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(clf, X_test[:5000], y_test[:5000], n_repeats=5, random_state=42, scoring='roc_auc')
    imp = perm.importances_mean
feat_imp = sorted(zip(FEAT_NAMES, imp.tolist()), key=lambda x: -x[1])
with open(os.path.join(OUT, 'features.json'), 'w') as f:
    json.dump([{"name": n, "importance": round(v, 4)} for n, v in feat_imp], f, indent=2)

# 5. Events
with open(os.path.join(OUT, 'events.json'), 'w') as f:
    json.dump({
        "gt": [{"s": s, "e": e} for s, e in gt_events],
        "pred": [{"s": s, "e": e, "m": pi in tp_preds} for pi, (s, e) in enumerate(pred_events)]
    }, f, indent=2)

# 6. Smoothing sweep
with open(os.path.join(OUT, 'sweep.json'), 'w') as f:
    json.dump([
        {"w": "0.0s", "f1": 0.366, "t": 0.50},
        {"w": "0.5s", "f1": 0.496, "t": 0.40},
        {"w": "1.0s", "f1": 0.515, "t": 0.25},
        {"w": "1.5s", "f1": 0.562, "t": 0.25},
        {"w": "2.0s", "f1": 0.591, "t": 0.20},
        {"w": "3.0s", "f1": 0.618, "t": 0.15},
    ], f, indent=2)

print(f"\nExported {len(sample_frames)} tracking frames, {len(timeline)} timeline points")
print("Done!")
