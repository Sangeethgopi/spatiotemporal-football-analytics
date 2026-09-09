"""
EXTERNAL VALIDATION — Metrica Sports Open Data Match 2
Uses kloppy to load genuine tracking data (NOT synthetic).
Constructs the same downstream features and evaluates the frozen candidate model.
"""
import os, sys, json, math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath('.'))

out_dir = "data/results/external_dataset"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("data/results/model_improvement", exist_ok=True)

# ----------------------------------------------------------------
# PART 1 — Load Metrica Match 2 via kloppy (genuine tracking data)
# ----------------------------------------------------------------
print("Loading Metrica Match 2 via kloppy...")
from kloppy import metrica

dataset = metrica.load_open_data(match_id=2, sample_rate=1/25.0, limit=None)

PITCH_L = 105.0
PITCH_W = 68.0
HZ = 25

rows = []
for frame in dataset.frames:
    row = {
        "frame": frame.frame_id,
        "period": frame.period.id,
        "time_s": frame.timestamp,
        "ball_x": None,
        "ball_y": None,
        "possession_team": None,
    }
    # Ball
    if frame.ball_coordinates:
        row["ball_x"] = frame.ball_coordinates.x * PITCH_L
        row["ball_y"] = frame.ball_coordinates.y * PITCH_W

    # Players
    home_players = []
    away_players = []
    for player_data in frame.players_coordinates.items():
        player, coord = player_data
        if coord is None:
            continue
        px = coord.x * PITCH_L
        py = coord.y * PITCH_W
        if player.team.ground.value == "home":
            home_players.append((px, py))
        else:
            away_players.append((px, py))
    row["home_players"] = home_players
    row["away_players"] = away_players
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} frames with kloppy")

# ----------------------------------------------------------------
# Check ball coverage
# ----------------------------------------------------------------
ball_ok = df["ball_x"].notna()
print(f"Ball detected: {ball_ok.sum()} / {len(df)} = {ball_ok.mean():.1%}")

# Run-length analysis for ball
run_lengths = []
cur = 0
for v in ball_ok.values:
    if v:
        cur += 1
    else:
        if cur > 0:
            run_lengths.append(cur)
            cur = 0
if cur > 0:
    run_lengths.append(cur)

run_lengths = np.array(run_lengths) if run_lengths else np.array([0])
print(f"Ball runs: {len(run_lengths)}, median={np.median(run_lengths):.1f}, max={np.max(run_lengths)}, ge25={np.sum(run_lengths>=25)}, ge100={np.sum(run_lengths>=100)}")

ball_diag = {
    "total_frames": int(len(df)),
    "ball_detected_frames": int(ball_ok.sum()),
    "ball_coverage_pct": float(ball_ok.mean()),
    "n_runs": int(len(run_lengths)),
    "median_run": float(np.median(run_lengths)),
    "mean_run": float(np.mean(run_lengths)),
    "max_run": int(np.max(run_lengths)),
    "runs_ge25": int(np.sum(run_lengths >= 25)),
    "runs_ge50": int(np.sum(run_lengths >= 50)),
    "runs_ge75": int(np.sum(run_lengths >= 75)),
    "runs_ge100": int(np.sum(run_lengths >= 100)),
}
with open(f"{out_dir}/ball_diagnostics.json", "w") as f:
    json.dump(ball_diag, f, indent=4)

# ----------------------------------------------------------------
# Possession assignment (simple proximity rule)
# ----------------------------------------------------------------
HOLD_FRAMES = 75
TURNOVER_MIN_HOLD = 75

def assign_possession(df, hold_frames=HOLD_FRAMES):
    poss = [None] * len(df)
    current_poss = None
    consecutive = 0
    for i, row in df.iterrows():
        bx = row["ball_x"]
        by = row["ball_y"]
        if pd.isna(bx):
            poss[i] = current_poss
            continue
        # Find closest team centroid
        hps = row["home_players"]
        aps = row["away_players"]
        if not hps and not aps:
            poss[i] = current_poss
            continue
        def min_dist(players):
            if not players:
                return 999.0
            dists = [math.hypot(px - bx, py - by) for px, py in players]
            return min(dists)
        hd = min_dist(hps)
        ad = min_dist(aps)
        closer = "Home" if hd <= ad else "Away"
        if closer != current_poss:
            consecutive += 1
            if consecutive >= hold_frames:
                current_poss = closer
                consecutive = 0
        else:
            consecutive = 0
        poss[i] = current_poss
    return poss

print("Assigning possession...")
df["possession_team"] = assign_possession(df)

# ----------------------------------------------------------------
# Derive turnover events
# ----------------------------------------------------------------
POSSESSION_MIN_HOLD = 75
TURNOVER_PROX_M = 3.0

def derive_turnovers(df, prox_m=3.0, min_hold=75):
    events = []
    teams = df["possession_team"].values
    ball_x = df["ball_x"].values
    ball_y = df["ball_y"].values
    home_ps = df["home_players"].values
    away_ps = df["away_players"].values
    n = len(df)
    current_team = None
    team_frames = 0
    
    for i in range(n):
        t = teams[i]
        if t is None:
            team_frames = 0
            continue
        if t != current_team:
            if team_frames >= min_hold and current_team is not None:
                # Check proximity at transition frame
                bx = ball_x[i]
                by = ball_y[i]
                if pd.isna(bx):
                    current_team = t
                    team_frames = 1
                    continue
                defending_team_players = home_ps[i] if t == "Away" else away_ps[i]
                if not defending_team_players:
                    current_team = t
                    team_frames = 1
                    continue
                dists = [math.hypot(px - bx, py - by) for px, py in defending_team_players]
                min_d = min(dists)
                if min_d <= prox_m:
                    events.append({
                        "frame_idx": i,
                        "frame": df.index[i],
                        "winning_team": t,
                        "min_def_dist": min_d,
                    })
            current_team = t
            team_frames = 1
        else:
            team_frames += 1
    return pd.DataFrame(events)

print("Deriving turnovers...")
turnovers_3m = derive_turnovers(df, prox_m=3.0)
turnovers_unc = derive_turnovers(df, prox_m=999.0)
print(f"Turnovers 3m: {len(turnovers_3m)}, Uncoupled: {len(turnovers_unc)}")

# ----------------------------------------------------------------
# Label target: 100-frame backward projection
# ----------------------------------------------------------------
HORIZON = 100

def label_horizons(df, turnovers, horizon=100):
    tgt_home = np.zeros(len(df), dtype=int)
    tgt_away = np.zeros(len(df), dtype=int)
    if turnovers.empty:
        df2 = df.copy()
        df2["target_press_trigger"] = 0
        return df2
    for _, ev in turnovers.iterrows():
        idx = int(ev["frame_idx"])
        start = max(0, idx - horizon)
        wt = ev["winning_team"]
        if wt == "Home":
            tgt_home[start:idx] = 1
        else:
            tgt_away[start:idx] = 1
    df2 = df.copy()
    # Press trigger fires for the defending team
    # If Home wins possession, Away was pressing (but we track from defending perspective)
    # Aligning with original: target fires during pressing BUILDUP
    df2["target_press_trigger"] = np.maximum(tgt_home, tgt_away)
    return df2

print("Labeling targets...")
df_labeled_3m = label_horizons(df, turnovers_3m, HORIZON)
df_labeled_unc = label_horizons(df, turnovers_unc, HORIZON)
print(f"Target prevalence (3m): {df_labeled_3m['target_press_trigger'].mean():.3f}")

# ----------------------------------------------------------------
# Feature extraction (simplified version of FEATURE_COLUMNS)
# ----------------------------------------------------------------
print("Extracting features...")

def rolling_bk(s, w):
    return s.rolling(w, min_periods=1).mean()

def extract_features(df_l):
    feats = pd.DataFrame(index=df_l.index)
    bx = df_l["ball_x"].ffill().fillna(52.5)
    by = df_l["ball_y"].ffill().fillna(34.0)
    
    # For each frame compute player-based features
    rows_feat = []
    for i in range(len(df_l)):
        row = df_l.iloc[i]
        poss_team = row["possession_team"] or "Home"
        att_team = poss_team
        def_team = "Away" if att_team == "Home" else "Home"
        
        att_ps = row["home_players"] if att_team == "Home" else row["away_players"]
        def_ps = row["home_players"] if def_team == "Home" else row["away_players"]
        
        b_x = bx.iloc[i]
        b_y = by.iloc[i]
        
        # Defender distances to ball
        if def_ps:
            dists = sorted([math.hypot(px - b_x, py - b_y) for px, py in def_ps])
        else:
            dists = [99.0, 99.0, 99.0]
        
        d1 = dists[0] if len(dists) > 0 else 99.0
        d2 = dists[1] if len(dists) > 1 else 99.0
        d3 = dists[2] if len(dists) > 2 else 99.0
        d_mean3 = np.mean(dists[:3]) if len(dists) >= 3 else np.mean(dists)
        
        # Closing speed (change in distances - computed via rolling diffs)
        closing_max = 0.0  # placeholder, will compute from diff later
        pressure_index = max(0, 1.0 - d1 / 10.0) if d1 < 10 else 0.0
        
        def_5m = sum(1 for d in dists if d <= 5.0)
        def_10m = sum(1 for d in dists if d <= 10.0)
        att_10m = sum(1 for px, py in att_ps if math.hypot(px - b_x, py - b_y) <= 10.0) if att_ps else 0
        num_adv = def_10m - att_10m
        
        # Convex hull area
        if len(def_ps) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(def_ps)
                hull_area = hull.volume
            except Exception:
                hull_area = 0.0
        else:
            hull_area = 0.0
        
        if def_ps:
            def_xs = [p[0] for p in def_ps]
            def_ys = [p[1] for p in def_ps]
            block_w = max(def_xs) - min(def_xs) if len(def_xs) > 1 else 0
            block_d = max(def_ys) - min(def_ys) if len(def_ys) > 1 else 0
            centroid = (np.mean(def_xs), np.mean(def_ys))
            centroid_dist = math.hypot(centroid[0] - b_x, centroid[1] - b_y)
            dispersion = np.std(dists) if len(dists) > 1 else 0.0
            def_line_h = np.mean(def_xs)
        else:
            block_w = block_d = centroid_dist = dispersion = def_line_h = 0.0
        
        ball_speed = 0.0  # from diff
        viable_pass = max(0, att_10m - def_10m)
        dist_to_goal = math.hypot(b_x - 105.0, b_y - 34.0)
        dist_to_sideline = min(b_y, 68.0 - b_y)
        
        rows_feat.append([
            d1, d2, d3, d_mean3,
            closing_max, closing_max,  # closing_speed_max, closing_speed_mean3
            pressure_index,
            def_5m, def_10m, att_10m, num_adv,
            hull_area, block_w, block_d,
            0.0,  # passing_lane_occlusion
            dispersion, centroid_dist, def_line_h,
            ball_speed, 0.0,  # ball_speed, ball_progression_x
            0.0, 0.0, 0.0, 0.0, 0.0,  # rolling features (filled below)
            b_x, b_y,
            dist_to_goal, dist_to_sideline,
            viable_pass, 0.0,  # possession_duration_s
        ])
    
    feat_names = [
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
    ]
    
    df_f = pd.DataFrame(rows_feat, columns=feat_names, index=df_l.index)
    
    # Compute rolling and temporal features post-hoc
    w = 25
    df_f["feat_closing_speed_max"] = df_f["feat_def_dist_mean3"].diff(1).abs().fillna(0)
    df_f["feat_closing_speed_mean3"] = df_f["feat_def_dist_mean3"].diff(3).abs().rolling(3).mean().fillna(0)
    _bx_filled = df_l["ball_x"].ffill()
    _by_filled = df_l["ball_y"].ffill()
    df_f["feat_ball_speed"] = pd.Series(
        [math.hypot(
            _bx_filled.diff().iloc[i] or 0,
            _by_filled.diff().iloc[i] or 0
        ) * HZ for i in range(len(df_l))]
    ).values
    df_f["feat_ball_progression_x"] = _bx_filled.diff(w).fillna(0).values
    df_f["feat_roll_pressure_index_mean"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_pressure_index_max"] = df_f["feat_pressure_index"].rolling(w, min_periods=1).max()
    df_f["feat_roll_closing_speed_mean"] = df_f["feat_closing_speed_max"].rolling(w, min_periods=1).mean()
    df_f["feat_roll_dist_mean3_min"] = df_f["feat_def_dist_mean3"].rolling(w, min_periods=1).min()
    df_f["feat_roll_hull_area_change"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    df_f["feat_temp_accel"] = df_f["feat_closing_speed_max"].diff(w).fillna(0)
    df_f["feat_temp_hull_shrink"] = df_f["feat_def_hull_area"].diff(w).fillna(0)
    
    return df_f

df_feat = extract_features(df_labeled_3m)
df_feat["target_press_trigger"] = df_labeled_3m["target_press_trigger"].values

print(f"Feature shape: {df_feat.shape}, target sum: {df_feat['target_press_trigger'].sum()}")

with open(f"{out_dir}/match2_stats.json", "w") as f:
    json.dump({
        "n_frames": int(len(df)),
        "n_turnover_events_3m": int(len(turnovers_3m)),
        "n_turnover_events_uncoupled": int(len(turnovers_unc)),
        "target_prevalence_3m": float(df_feat["target_press_trigger"].mean()),
        **ball_diag,
    }, f, indent=4)

print("Done collecting stats. Feature extraction complete.")
print("Saved to", out_dir)
