"""
FULL EXTERNAL VALIDATION on Metrica Sports Open Data Match 2
- Genuine, non-synthetic broadcast-tracked football data
- 141,156 frames, 25fps, ~48 minutes
- FROZEN candidate model evaluated on this external data
"""
import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from scripts.run_apples_apples import evaluate_events_strict, evaluate_frame_metrics
from kloppy import metrica

PITCH_L = 105.0
PITCH_W = 68.0
HZ = 25
out_dir = "data/results/external_dataset"
os.makedirs(out_dir, exist_ok=True)

# ================================================================
# A. Load internal synthetic data & train frozen candidate model
# ================================================================
print("Training frozen candidate model on synthetic Match 1...")
paths = ensure_dataset_available('data', 'match_1')
df_raw_m1 = load_match_tracking(paths['home'], paths['away'])
df_poss_m1 = assign_frame_possession(df_raw_m1)
turnovers_m1 = derive_turnover_events(df_poss_m1)
df_labeled_m1 = label_turnover_horizons(df_poss_m1, turnovers_m1)
df_feat_m1 = extract_pressing_features(df_labeled_m1)
n = len(df_feat_m1)
split_idx = int(n * 0.70)

df_feat_m1['feat_temp_accel'] = df_feat_m1['feat_closing_speed_max'].diff(25).fillna(0)
df_feat_m1['feat_temp_hull_shrink'] = df_feat_m1['feat_def_hull_area'].diff(25).fillna(0)
best_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']

df_train_full = df_feat_m1.iloc[:split_idx - 100]
clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf.fit(df_train_full[best_cols].fillna(0), df_train_full['target_press_trigger'].values)
print("Model trained.")

# ================================================================
# B. Load Metrica Match 2 (genuine external tracking data)
# ================================================================
print("Loading Metrica Match 2 (genuine tracking)...")
dataset = metrica.load_open_data(match_id=2, sample_rate=None, limit=None)

rows = []
for frame in dataset.frames:
    row = {"frame": frame.frame_id, "ball_x": None, "ball_y": None,
           "home_players": [], "away_players": []}
    if frame.ball_coordinates:
        row["ball_x"] = frame.ball_coordinates.x * PITCH_L
        row["ball_y"] = frame.ball_coordinates.y * PITCH_W
    for player, coord in frame.players_coordinates.items():
        if coord is None: continue
        px = coord.x * PITCH_L; py = coord.y * PITCH_W
        if player.team.ground.value == "home":
            row["home_players"].append((px, py))
        else:
            row["away_players"].append((px, py))
    rows.append(row)

df_m2 = pd.DataFrame(rows)
print(f"Match 2 frames: {len(df_m2)}")

# ================================================================
# Ball continuity stats
# ================================================================
ball_ok = df_m2["ball_x"].notna()
run_lengths = []
cur = 0
for v in ball_ok.values:
    if v: cur += 1
    else:
        if cur > 0: run_lengths.append(cur); cur = 0
if cur > 0: run_lengths.append(cur)
run_lengths = np.array(run_lengths) if run_lengths else np.array([0])

m2_ball_diag = {
    "total_frames": int(len(df_m2)),
    "ball_detected": int(ball_ok.sum()),
    "ball_coverage_pct": float(ball_ok.mean()),
    "n_runs": int(len(run_lengths)),
    "median_run": float(np.median(run_lengths)),
    "mean_run": float(np.mean(run_lengths)),
    "max_run": int(np.max(run_lengths)),
    "runs_ge25": int(np.sum(run_lengths >= 25)),
    "runs_ge100": int(np.sum(run_lengths >= 100)),
}
print(f"Ball coverage: {ball_ok.mean():.1%}, median_run={np.median(run_lengths):.1f}, n_runs={len(run_lengths)}")

# ================================================================
# Possession assignment
# ================================================================
HOLD = 75

def assign_possession_m2(df, hold_frames=HOLD):
    poss = [None] * len(df)
    current_poss = None
    consecutive = 0
    for i, row in df.iterrows():
        bx = row["ball_x"]; by = row["ball_y"]
        if pd.isna(bx) or pd.isna(by):
            poss[i] = current_poss; continue
        hps = row["home_players"]; aps = row["away_players"]
        if not hps or not aps:
            poss[i] = current_poss; continue
        hd = min(math.hypot(px-bx, py-by) for px, py in hps)
        ad = min(math.hypot(px-bx, py-by) for px, py in aps)
        closer = "Home" if hd <= ad else "Away"
        if closer != current_poss:
            consecutive += 1
            if consecutive >= hold_frames:
                current_poss = closer; consecutive = 0
        else:
            consecutive = 0
        poss[i] = current_poss
    return poss

print("Assigning possession...")
df_m2["possession_team"] = assign_possession_m2(df_m2)
poss_arr = np.array(df_m2["possession_team"])
print(f"Home: {np.sum(poss_arr=='Home')}, Away: {np.sum(poss_arr=='Away')}, None: {np.sum(poss_arr==None)}")

# ================================================================
# Derive turnovers
# ================================================================
def derive_turnovers_m2(df, prox_m=3.0, min_hold=75):
    events = []
    teams = df["possession_team"].values
    ball_x = df["ball_x"].values
    ball_y = df["ball_y"].values
    home_ps = df["home_players"].values
    away_ps = df["away_players"].values
    n = len(df)
    current_team = None; team_frames = 0
    for i in range(n):
        t = teams[i]
        if t is None: team_frames = 0; continue
        if t != current_team:
            if team_frames >= min_hold and current_team is not None:
                bx = ball_x[i]; by = ball_y[i]
                if not pd.isna(bx):
                    def_ps = home_ps[i] if t == "Away" else away_ps[i]
                    if def_ps:
                        dists = [math.hypot(px-bx, py-by) for px, py in def_ps]
                        if prox_m >= 999 or min(dists) <= prox_m:
                            events.append({"frame_idx": i, "frame": df.index[i], "winning_team": t})
            current_team = t; team_frames = 1
        else:
            team_frames += 1
    return pd.DataFrame(events)

print("Deriving turnovers...")
turnovers_m2_3m = derive_turnovers_m2(df_m2, prox_m=3.0)
turnovers_m2_unc = derive_turnovers_m2(df_m2, prox_m=999.0)
print(f"Turnovers (3m): {len(turnovers_m2_3m)}, Uncoupled: {len(turnovers_m2_unc)}")

# ================================================================
# Label target
# ================================================================
def label_target_m2(df, turnovers, horizon=100):
    tgt = np.zeros(len(df), dtype=int)
    for _, ev in turnovers.iterrows():
        idx = int(ev["frame_idx"])
        start = max(0, idx - horizon)
        tgt[start:idx] = 1
    return tgt

print("Labeling targets...")
y_m2_3m = label_target_m2(df_m2, turnovers_m2_3m)
y_m2_unc = label_target_m2(df_m2, turnovers_m2_unc)
print(f"Positive frames (3m): {y_m2_3m.sum()}, prevalence: {y_m2_3m.mean():.4f}")

# ================================================================
# Extract features for Match 2
# ================================================================
print("Extracting features (this takes a few minutes)...")

from scipy.spatial import ConvexHull

def extract_m2_features(df_l, y_target):
    rows_feat = []
    bx_filled = df_l["ball_x"].ffill().fillna(52.5)
    by_filled = df_l["ball_y"].ffill().fillna(34.0)
    poss_teams = df_l["possession_team"].values
    
    for i in range(len(df_l)):
        row = df_l.iloc[i]
        poss_team = poss_teams[i] or "Home"
        att_team = poss_team
        def_team = "Away" if att_team == "Home" else "Home"
        att_ps = row["home_players"] if att_team == "Home" else row["away_players"]
        def_ps = row["home_players"] if def_team == "Home" else row["away_players"]
        b_x = bx_filled.iloc[i]; b_y = by_filled.iloc[i]
        
        if def_ps:
            dists = sorted([math.hypot(px-b_x, py-b_y) for px, py in def_ps])
        else:
            dists = [99.0]
        
        d1 = dists[0] if len(dists)>0 else 99.0
        d2 = dists[1] if len(dists)>1 else 99.0
        d3 = dists[2] if len(dists)>2 else 99.0
        d_m3 = np.mean(dists[:3]) if len(dists)>=3 else np.mean(dists)
        pressure_index = max(0, 1.0 - d1/10.0) if d1 < 10 else 0.0
        def_5m = sum(1 for d in dists if d <= 5)
        def_10m = sum(1 for d in dists if d <= 10)
        att_10m = sum(1 for px,py in att_ps if math.hypot(px-b_x,py-b_y)<=10) if att_ps else 0
        
        hull_area = 0.0
        if len(def_ps) >= 3:
            try:
                h = ConvexHull(def_ps)
                hull_area = h.volume
            except: pass
        
        def_xs = [p[0] for p in def_ps] if def_ps else [0]
        def_ys = [p[1] for p in def_ps] if def_ps else [0]
        block_w = max(def_xs)-min(def_xs) if len(def_xs)>1 else 0
        block_d = max(def_ys)-min(def_ys) if len(def_ys)>1 else 0
        centroid = (np.mean(def_xs), np.mean(def_ys))
        centroid_dist = math.hypot(centroid[0]-b_x, centroid[1]-b_y)
        dispersion = np.std(dists) if len(dists)>1 else 0
        def_line_h = np.mean(def_xs)
        viable_pass = max(0, att_10m - def_10m)
        dist_to_goal = math.hypot(b_x - 105.0, b_y - 34.0)
        dist_to_sideline = min(b_y, 68.0 - b_y)
        
        rows_feat.append([
            d1, d2, d3, d_m3, 0, 0, pressure_index,
            def_5m, def_10m, att_10m, def_10m - att_10m,
            hull_area, block_w, block_d, 0.0,
            dispersion, centroid_dist, def_line_h,
            0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            b_x, b_y, dist_to_goal, dist_to_sideline, viable_pass, 0.0
        ])
    
    feat_names = [
        "feat_def_dist_1","feat_def_dist_2","feat_def_dist_3","feat_def_dist_mean3",
        "feat_closing_speed_max","feat_closing_speed_mean3","feat_pressure_index",
        "feat_def_density_5m","feat_def_density_10m","feat_att_density_10m","feat_numerical_advantage_10m",
        "feat_def_hull_area","feat_def_block_width","feat_def_block_depth","feat_passing_lane_occlusion",
        "feat_def_dispersion","feat_def_centroid_dist_to_ball","feat_def_line_height",
        "feat_ball_speed","feat_ball_progression_x",
        "feat_roll_pressure_index_mean","feat_roll_pressure_index_max","feat_roll_closing_speed_mean",
        "feat_roll_dist_mean3_min","feat_roll_hull_area_change",
        "feat_ball_x","feat_ball_y","feat_dist_to_goal","feat_dist_to_sideline",
        "feat_viable_passing_options","feat_possession_duration_s",
    ]
    df_f = pd.DataFrame(rows_feat, columns=feat_names, index=df_l.index)
    
    w = 25
    df_f["feat_closing_speed_max"] = df_f["feat_def_dist_mean3"].diff(1).abs().fillna(0)
    df_f["feat_closing_speed_mean3"] = df_f["feat_def_dist_mean3"].diff(3).abs().rolling(3).mean().fillna(0)
    df_f["feat_ball_speed"] = bx_filled.diff().abs().fillna(0).values * HZ
    df_f["feat_ball_progression_x"] = bx_filled.diff(w).fillna(0).values
    df_f["feat_roll_pressure_index_mean"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_pressure_index_max"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).max()
    df_f["feat_roll_closing_speed_mean"] = df_f["feat_closing_speed_max"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_dist_mean3_min"] = df_f["feat_def_dist_mean3"].rolling(w, min_periods=1).min()
    df_f["feat_roll_hull_area_change"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    df_f["feat_temp_accel"] = df_f["feat_closing_speed_max"].diff(w).fillna(0)
    df_f["feat_temp_hull_shrink"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    df_f["target_press_trigger"] = y_target
    return df_f

df_m2_feat = extract_m2_features(df_m2, y_m2_3m)
print(f"Match 2 feature shape: {df_m2_feat.shape}, target sum: {df_m2_feat['target_press_trigger'].sum()}")

# ================================================================
# Evaluate frozen model on Metrica Match 2
# ================================================================
print("Evaluating frozen model on Match 2...")
X_m2 = df_m2_feat[best_cols].fillna(0)
p_m2_raw = clf.predict_proba(X_m2)[:, 1]
p_m2_smooth = pd.Series(p_m2_raw).rolling(25, min_periods=1, center=False).mean().values
y_m2 = df_m2_feat["target_press_trigger"].values

ev_m2 = evaluate_events_strict(y_m2, p_m2_smooth, 0.50)
fm_m2 = evaluate_frame_metrics(y_m2, p_m2_smooth, 0.50)
ev_m2_unc = evaluate_events_strict(label_target_m2(df_m2, turnovers_m2_unc), p_m2_smooth, 0.50)

# ================================================================
# Save results
# ================================================================
results = {
    "dataset": "Metrica Sports Open Data Match 2 (genuine, non-synthetic)",
    "n_frames": int(len(df_m2)),
    "n_turnover_events_3m": int(len(turnovers_m2_3m)),
    "n_turnover_events_uncoupled": int(len(turnovers_m2_unc)),
    "target_prevalence": float(y_m2.mean()),
    "ball_diagnostics": m2_ball_diag,
    "model_evaluation": {
        "roc_auc": fm_m2["roc"],
        "pr_auc": fm_m2["pr"],
        "frame_precision": fm_m2["f_prec"],
        "frame_recall": fm_m2["f_rec"],
        "frame_f1": fm_m2["f_f1"],
        "event_precision": ev_m2["prec"],
        "event_recall": ev_m2["rec"],
        "event_f1": ev_m2["f1"],
        "tp": ev_m2["tp"],
        "fp": ev_m2["fp"],
        "fn": ev_m2["fn"],
        "n_pred": ev_m2["n_pred"],
        "n_gt": ev_m2["n_gt"],
    },
    "uncoupled_evaluation": {
        "event_f1": ev_m2_unc["f1"],
        "precision": ev_m2_unc["prec"],
        "recall": ev_m2_unc["rec"],
        "tp": ev_m2_unc["tp"],
        "fp": ev_m2_unc["fp"],
        "fn": ev_m2_unc["fn"],
        "n_gt": ev_m2_unc["n_gt"],
    }
}

for d in [ev_m2, ev_m2_unc, results["model_evaluation"], results["uncoupled_evaluation"]]:
    d.pop("pred_events", None)
    d.pop("true_events", None)

with open(f"{out_dir}/EXTERNAL_VALIDATION.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n=== EXTERNAL VALIDATION RESULTS ===")
print(f"Turnovers (3m): {len(turnovers_m2_3m)}, Uncoupled: {len(turnovers_m2_unc)}")
print(f"Event F1: {ev_m2['f1']:.3f}, Precision: {ev_m2['prec']:.3f}, Recall: {ev_m2['rec']:.3f}")
print(f"TP: {ev_m2['tp']}, FP: {ev_m2['fp']}, FN: {ev_m2['fn']}")
print(f"GT Events: {ev_m2['n_gt']}, Pred Events: {ev_m2['n_pred']}")
print(f"ROC-AUC: {fm_m2['roc']:.3f}, PR-AUC: {fm_m2['pr']:.3f}")
print(f"Uncoupled F1: {ev_m2_unc['f1']:.3f}")
print("Saved to", out_dir)
