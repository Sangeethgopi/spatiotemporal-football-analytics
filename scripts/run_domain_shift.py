"""
Domain shift analysis: check probability distribution on Match 2 vs threshold sweep
"""
import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from scripts.run_apples_apples import evaluate_events_strict
from kloppy import metrica
from scipy.spatial import ConvexHull

PITCH_L = 105.0; PITCH_W = 68.0; HZ = 25
out_dir = "data/results/external_dataset"

# ---- Train frozen model ----
print("Training model on synthetic Match 1...")
paths = ensure_dataset_available('data', 'match_1')
df_raw = load_match_tracking(paths['home'], paths['away'])
df_poss = assign_frame_possession(df_raw)
turnovers = derive_turnover_events(df_poss)
df_labeled = label_turnover_horizons(df_poss, turnovers)
df_feat = extract_pressing_features(df_labeled)
n = len(df_feat); split_idx = int(n * 0.70)
df_feat['feat_temp_accel'] = df_feat['feat_closing_speed_max'].diff(25).fillna(0)
df_feat['feat_temp_hull_shrink'] = df_feat['feat_def_hull_area'].diff(25).fillna(0)
best_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']
df_train = df_feat.iloc[:split_idx - 100]
clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
clf.fit(df_train[best_cols].fillna(0), df_train['target_press_trigger'].values)

# ---- Load Match 2 ----
print("Loading Match 2...")
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

def assign_possession_m2(df, hold_frames=75):
    poss = [None]*len(df)
    current_poss = None; consecutive = 0
    for i, row in df.iterrows():
        bx = row["ball_x"]; by = row["ball_y"]
        if pd.isna(bx) or pd.isna(by): poss[i] = current_poss; continue
        hps = row["home_players"]; aps = row["away_players"]
        if not hps or not aps: poss[i] = current_poss; continue
        hd = min(math.hypot(px-bx, py-by) for px, py in hps)
        ad = min(math.hypot(px-bx, py-by) for px, py in aps)
        closer = "Home" if hd <= ad else "Away"
        if closer != current_poss:
            consecutive += 1
            if consecutive >= hold_frames: current_poss = closer; consecutive = 0
        else: consecutive = 0
        poss[i] = current_poss
    return poss

df_m2["possession_team"] = assign_possession_m2(df_m2)

def extract_m2_features(df_l):
    rows_feat = []
    bx_f = df_l["ball_x"].ffill().fillna(52.5)
    by_f = df_l["ball_y"].ffill().fillna(34.0)
    poss_teams = df_l["possession_team"].values
    for i in range(len(df_l)):
        row = df_l.iloc[i]
        poss_team = poss_teams[i] or "Home"
        def_team = "Away" if poss_team == "Home" else "Home"
        att_ps = row["home_players"] if poss_team == "Home" else row["away_players"]
        def_ps = row["home_players"] if def_team == "Home" else row["away_players"]
        b_x = bx_f.iloc[i]; b_y = by_f.iloc[i]
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
            try: hull_area = ConvexHull(def_ps).volume
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
        rows_feat.append([d1,d2,d3,d_m3,0,0,pressure_index,def_5m,def_10m,att_10m,
                          def_10m-att_10m,hull_area,block_w,block_d,0,dispersion,
                          centroid_dist,def_line_h,0,0,0,0,0,0,0,b_x,b_y,
                          dist_to_goal,dist_to_sideline,viable_pass,0])
    feat_names = FEATURE_COLUMNS
    df_f = pd.DataFrame(rows_feat, columns=feat_names, index=df_l.index)
    w = 25
    df_f["feat_closing_speed_max"] = df_f["feat_def_dist_mean3"].diff(1).abs().fillna(0)
    df_f["feat_closing_speed_mean3"] = df_f["feat_def_dist_mean3"].diff(3).abs().rolling(3).mean().fillna(0)
    df_f["feat_ball_speed"] = bx_f.diff().abs().fillna(0).values * HZ
    df_f["feat_ball_progression_x"] = bx_f.diff(w).fillna(0).values
    df_f["feat_roll_pressure_index_mean"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_pressure_index_max"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).max()
    df_f["feat_roll_closing_speed_mean"] = df_f["feat_closing_speed_max"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_dist_mean3_min"] = df_f["feat_def_dist_mean3"].rolling(w, min_periods=1).min()
    df_f["feat_roll_hull_area_change"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    df_f["feat_temp_accel"] = df_f["feat_closing_speed_max"].diff(w).fillna(0)
    df_f["feat_temp_hull_shrink"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    return df_f

print("Extracting features...")
df_m2_feat = extract_m2_features(df_m2)

def label_target_m2(df, turnovers_df, horizon=100):
    tgt = np.zeros(len(df), dtype=int)
    for _, ev in turnovers_df.iterrows():
        idx = int(ev["frame_idx"])
        start = max(0, idx - horizon)
        tgt[start:idx] = 1
    return tgt

def derive_turnovers_m2(df, prox_m=3.0, min_hold=75):
    events = []
    teams = df["possession_team"].values
    ball_x = df["ball_x"].values; ball_y = df["ball_y"].values
    home_ps = df["home_players"].values; away_ps = df["away_players"].values
    current_team = None; team_frames = 0
    for i in range(len(df)):
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
        else: team_frames += 1
    return pd.DataFrame(events)

turnovers_m2 = derive_turnovers_m2(df_m2, prox_m=3.0)
y_m2 = label_target_m2(df_m2, turnovers_m2)

# ---- Probability distribution analysis ----
print("Analysing probability distributions...")
X_m2 = df_m2_feat[best_cols].fillna(0)
X_m1_train = df_train[best_cols].fillna(0)

p_m2 = clf.predict_proba(X_m2)[:, 1]
p_m1_test = clf.predict_proba(df_feat.iloc[split_idx:][best_cols].fillna(0))[:, 1]
p_m1_smooth = pd.Series(p_m1_test).rolling(25, min_periods=1, center=False).mean().values
p_m2_smooth = pd.Series(p_m2).rolling(25, min_periods=1, center=False).mean().values

print(f"\nM1 (synthetic test) prob distribution:")
print(f"  Mean: {np.mean(p_m1_smooth):.4f}, Std: {np.std(p_m1_smooth):.4f}")
print(f"  P95: {np.percentile(p_m1_smooth, 95):.4f}, P99: {np.percentile(p_m1_smooth, 99):.4f}")
print(f"  Fraction > 0.50: {np.mean(p_m1_smooth > 0.50):.4f}")

print(f"\nM2 (real external) prob distribution:")
print(f"  Mean: {np.mean(p_m2_smooth):.4f}, Std: {np.std(p_m2_smooth):.4f}")
print(f"  P95: {np.percentile(p_m2_smooth, 95):.4f}, P99: {np.percentile(p_m2_smooth, 99):.4f}")
print(f"  Fraction > 0.50: {np.mean(p_m2_smooth > 0.50):.4f}")

# Threshold sweep on M2
print("\nThreshold sweep on Match 2:")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
thresh_results = []
for t in thresholds:
    ev = evaluate_events_strict(y_m2, p_m2_smooth, t)
    ev.pop("pred_events", None); ev.pop("true_events", None)
    thresh_results.append({"threshold": t, **ev})
    print(f"  t={t:.2f}: F1={ev['f1']:.3f} P={ev['prec']:.3f} R={ev['rec']:.3f} TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} preds={ev['n_pred']}")

# Feature distribution comparison (key features)
key_feats = ["feat_def_dist_1", "feat_pressure_index", "feat_def_hull_area", "feat_closing_speed_max"]
feat_dists = {}
for f in key_feats:
    m1_vals = df_feat.iloc[split_idx:][f].fillna(0).values
    m2_vals = df_m2_feat[f].fillna(0).values
    feat_dists[f] = {
        "m1_mean": float(np.mean(m1_vals)), "m1_std": float(np.std(m1_vals)),
        "m1_p50": float(np.median(m1_vals)), "m1_p95": float(np.percentile(m1_vals, 95)),
        "m2_mean": float(np.mean(m2_vals)), "m2_std": float(np.std(m2_vals)),
        "m2_p50": float(np.median(m2_vals)), "m2_p95": float(np.percentile(m2_vals, 95)),
    }

domain_shift = {
    "m1_prob_mean": float(np.mean(p_m1_smooth)),
    "m1_prob_std": float(np.std(p_m1_smooth)),
    "m1_prob_p95": float(np.percentile(p_m1_smooth, 95)),
    "m1_prob_p99": float(np.percentile(p_m1_smooth, 99)),
    "m1_frac_above_050": float(np.mean(p_m1_smooth > 0.50)),
    "m2_prob_mean": float(np.mean(p_m2_smooth)),
    "m2_prob_std": float(np.std(p_m2_smooth)),
    "m2_prob_p95": float(np.percentile(p_m2_smooth, 95)),
    "m2_prob_p99": float(np.percentile(p_m2_smooth, 99)),
    "m2_frac_above_050": float(np.mean(p_m2_smooth > 0.50)),
    "threshold_sweep": thresh_results,
    "feature_distributions": feat_dists,
}

with open(f"{out_dir}/DOMAIN_SHIFT_ANALYSIS.json", "w") as f:
    json.dump(domain_shift, f, indent=4)

print("\nSaved domain shift analysis.")
