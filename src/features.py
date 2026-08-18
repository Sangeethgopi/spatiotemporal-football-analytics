"""
Spatial-Temporal Feature Engineering Module for Pressing Intensity.
Extracts physically bounded, strictly causal kinematic and geometric features from tracking coordinates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.spatial import ConvexHull
from src.config import (
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    SAMPLING_RATE_HZ,
    FEATURE_LOOKBACK_FRAMES,
    MAX_REALISTIC_PLAYER_SPEED_M_S
)
from src.data_loader import get_player_columns


def compute_convex_hull_area(points: np.ndarray) -> float:
    """
    Computes 2D convex hull spatial area occupied by players in m^2.
    """
    valid_pts = points[~np.isnan(points).any(axis=1)]
    if len(valid_pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(valid_pts)
        return float(hull.volume)
    except Exception:
        dx = np.ptp(valid_pts[:, 0])
        dy = np.ptp(valid_pts[:, 1])
        return float(0.65 * dx * dy)


def compute_passing_lane_occlusion(
    bx: float, by: float,
    att_x: np.ndarray, att_y: np.ndarray,
    def_x: np.ndarray, def_y: np.ndarray,
    occlusion_radius: float = 2.0
) -> Tuple[float, int]:
    """
    Computes fraction of passing lanes (ball to teammates) occluded by defenders,
    and returns the absolute number of viable (unoccluded) passing options.
    A lane is occluded if a defender is within `occlusion_radius` of the pass trajectory.
    """
    valid_att = ~(np.isnan(att_x) | np.isnan(att_y))
    valid_def = ~(np.isnan(def_x) | np.isnan(def_y))
    
    if not np.any(valid_att) or not np.any(valid_def):
        return 0.0, 0
        
    ax, ay = att_x[valid_att], att_y[valid_att]
    dx, dy = def_x[valid_def], def_y[valid_def]
    
    # Vectors to teammates
    vx, vy = ax - bx, ay - by
    dist_att = np.sqrt(vx**2 + vy**2)
    
    # Filter teammates too close (e.g. ball carrier themselves)
    valid_receivers = dist_att > 1.0
    if not np.any(valid_receivers):
        return 0.0
        
    vx, vy = vx[valid_receivers], vy[valid_receivers]
    dist_att = dist_att[valid_receivers]
    ux, uy = vx / dist_att, vy / dist_att  # Unit vectors to receivers
    
    occluded_lanes = 0
    # For each receiver, check if any defender blocks the lane
    for i in range(len(ux)):
        # Defender vectors from ball
        dwx, dwy = dx - bx, dy - by
        # Projection of defenders onto passing lane
        proj = dwx * ux[i] + dwy * uy[i]
        
        # Defender must be between ball and receiver
        in_between = (proj > 0) & (proj < dist_att[i])
        
        # Perpendicular distance to passing lane squared
        perp_sq = (dwx**2 + dwy**2) - proj**2
        perp_sq = np.maximum(0.0, perp_sq) # avoid floating point negatives
        
        # Occluded if within radius
        is_occluded = in_between & (perp_sq < occlusion_radius**2)
        if np.any(is_occluded):
            occluded_lanes += 1
            
    return float(occluded_lanes / len(ux)), int(len(ux) - occluded_lanes)



def extract_pressing_features(df_possession: pd.DataFrame) -> pd.DataFrame:
    """
    Computes frame-level and rolling pressing features for the defending team
    relative to the attacking ball carrier.
    STRICTLY CAUSAL: Uses only past/current tracking information (<= t).
    """
    df = df_possession.copy()
    n_frames = len(df)

    home_players = get_player_columns(df, "Home")
    away_players = get_player_columns(df, "Away")

    # Arrays for computed features
    feat_def_dist_1 = np.full(n_frames, np.nan)
    feat_def_dist_2 = np.full(n_frames, np.nan)
    feat_def_dist_3 = np.full(n_frames, np.nan)
    feat_def_dist_mean3 = np.full(n_frames, np.nan)

    feat_closing_speed_max = np.full(n_frames, 0.0)
    feat_closing_speed_mean3 = np.full(n_frames, 0.0)
    feat_pressure_index = np.full(n_frames, 0.0)

    feat_def_density_5m = np.zeros(n_frames)
    feat_def_density_10m = np.zeros(n_frames)
    feat_att_density_10m = np.zeros(n_frames)
    feat_numerical_advantage_10m = np.zeros(n_frames)

    feat_def_hull_area = np.full(n_frames, np.nan)
    feat_def_block_width = np.full(n_frames, np.nan)
    feat_def_block_depth = np.full(n_frames, np.nan)
    feat_def_dispersion = np.full(n_frames, np.nan)
    feat_def_centroid_dist_to_ball = np.full(n_frames, np.nan)
    feat_def_line_height = np.full(n_frames, np.nan)

    # Pitch Location
    feat_ball_x = np.zeros(n_frames)
    feat_ball_y = np.zeros(n_frames)
    feat_dist_to_goal = np.zeros(n_frames)
    feat_dist_to_sideline = np.zeros(n_frames)
    
    feat_passing_lane_occlusion = np.full(n_frames, 0.0)
    feat_viable_passing_options = np.zeros(n_frames)
    feat_possession_duration_s = np.zeros(n_frames)
    
    feat_ball_speed = np.full(n_frames, 0.0)
    feat_ball_progression_x = np.full(n_frames, 0.0)

    ball_x_arr = df["Ball_x"].values
    ball_y_arr = df["Ball_y"].values
    ball_speed_arr = df.get("Ball_speed", pd.Series(np.zeros(n_frames))).values

    # Pre-extract player coordinates into numpy matrices for vectorized speed
    def get_team_matrix(pdict):
        pids = list(pdict.keys())
        x_mat = np.column_stack([df[pdict[p][0]].values for p in pids])
        y_mat = np.column_stack([df[pdict[p][1]].values for p in pids])
        vx_mat = np.column_stack([df.get(f"{p}_vx", pd.Series(np.zeros(n_frames))).values for p in pids])
        vy_mat = np.column_stack([df.get(f"{p}_vy", pd.Series(np.zeros(n_frames))).values for p in pids])
        return pids, x_mat, y_mat, vx_mat, vy_mat

    h_pids, h_x, h_y, h_vx, h_vy = get_team_matrix(home_players)
    a_pids, a_x, a_y, a_vx, a_vy = get_team_matrix(away_players)

    poss_team_arr = df["possession_team"].values

    for t in range(n_frames):
        poss = poss_team_arr[t]
        bx, by = ball_x_arr[t], ball_y_arr[t]

        if np.isnan(bx) or np.isnan(by) or poss not in ("Home", "Away"):
            continue

        if poss == "Home":
            # Defending team is Away, Attacking is Home
            def_x, def_y, def_vx, def_vy = a_x[t], a_y[t], a_vx[t], a_vy[t]
            att_x, att_y = h_x[t], h_y[t]
            defending_direction = -1.0
        else:
            # Defending team is Home, Attacking is Away
            def_x, def_y, def_vx, def_vy = h_x[t], h_y[t], h_vx[t], h_vy[t]
            att_x, att_y = a_x[t], a_y[t]
            defending_direction = 1.0

        # Valid non-NaN defenders
        valid_def = ~(np.isnan(def_x) | np.isnan(def_y))
        valid_att = ~(np.isnan(att_x) | np.isnan(att_y))

        if not np.any(valid_def):
            continue

        d_x, d_y = def_x[valid_def], def_y[valid_def]
        d_vx, d_vy = def_vx[valid_def], def_vy[valid_def]

        # Euclidean distances to ball carrier
        dist_to_ball = np.sqrt((d_x - bx)**2 + (d_y - by)**2)
        sorted_indices = np.argsort(dist_to_ball)
        sorted_dists = dist_to_ball[sorted_indices]

        # Proximity features (meters)
        d1 = sorted_dists[0] if len(sorted_dists) > 0 else np.nan
        d2 = sorted_dists[1] if len(sorted_dists) > 1 else np.nan
        d3 = sorted_dists[2] if len(sorted_dists) > 2 else np.nan

        feat_def_dist_1[t] = d1
        feat_def_dist_2[t] = d2
        feat_def_dist_3[t] = d3
        feat_def_dist_mean3[t] = np.mean(sorted_dists[:min(3, len(sorted_dists))])

        # Closing velocities towards ball: dot product of defender velocity vector and unit vector to ball
        dx_vec = bx - d_x
        dy_vec = by - d_y
        dist_safe = np.maximum(dist_to_ball, 0.2)
        unit_x = dx_vec / dist_safe
        unit_y = dy_vec / dist_safe

        # Compute dot product and clip strictly to physical human sprint speed (<= 10.5 m/s)
        closing_speeds = np.clip(d_vx * unit_x + d_vy * unit_y, -MAX_REALISTIC_PLAYER_SPEED_M_S, MAX_REALISTIC_PLAYER_SPEED_M_S)
        sorted_closing = closing_speeds[sorted_indices]

        c_max = float(np.max(closing_speeds)) if len(closing_speeds) > 0 else 0.0
        c_mean3 = float(np.mean(sorted_closing[:min(3, len(sorted_closing))])) if len(sorted_closing) > 0 else 0.0

        feat_closing_speed_max[t] = max(0.0, c_max)
        feat_closing_speed_mean3[t] = max(0.0, c_mean3)

        # Standardized Pressure Index formula:
        # P_index = Sum_{k=1..3} (1 / max(d_k, 1.0)) * (max(0, v_close_k) + 0.5)
        # Scaled to [0.00, 1.00] range using smooth sigmoid-based normalization
        raw_p_index = 0.0
        for k in range(min(3, len(sorted_dists))):
            dist_w = 1.0 / np.maximum(sorted_dists[k], 1.0)
            speed_w = np.maximum(0.0, sorted_closing[k]) + 0.5
            raw_p_index += dist_w * speed_w

        # Normalized to 0-1 scale: 3 defenders at 1m with 3m/s closing speed yields ~1.0
        norm_p_index = float(1.0 - np.exp(-raw_p_index / 3.0))
        feat_pressure_index[t] = norm_p_index

        # Density in 5m and 10m tactical radii
        def_in_5 = np.sum(dist_to_ball <= 5.0)
        def_in_10 = np.sum(dist_to_ball <= 10.0)

        att_dist_to_ball = np.sqrt((att_x[valid_att] - bx)**2 + (att_y[valid_att] - by)**2)
        att_in_10 = np.sum(att_dist_to_ball <= 10.0)

        feat_def_density_5m[t] = def_in_5
        feat_def_density_10m[t] = def_in_10
        feat_att_density_10m[t] = att_in_10
        feat_numerical_advantage_10m[t] = def_in_10 - att_in_10

        # Defensive Line Height & Convex Hull Compactness
        def_pts = np.column_stack([d_x, d_y])
        hull_area = compute_convex_hull_area(def_pts)
        c_def_x, c_def_y = np.mean(d_x), np.mean(d_y)
        disp = np.sqrt(np.std(d_x)**2 + np.std(d_y)**2)
        
        block_depth = np.ptp(d_x) if len(d_x) > 0 else np.nan
        block_width = np.ptp(d_y) if len(d_y) > 0 else np.nan

        feat_def_hull_area[t] = hull_area
        feat_def_block_width[t] = block_width
        feat_def_block_depth[t] = block_depth
        feat_def_dispersion[t] = disp
        feat_def_centroid_dist_to_ball[t] = np.sqrt((c_def_x - bx)**2 + (c_def_y - by)**2)
        feat_def_line_height[t] = np.max(d_x) if defending_direction == 1.0 else (PITCH_LENGTH_METERS - np.min(d_x))

        # Passing Lane Occlusion (Simplified Pitch Control Proxy)
        occ_pct, viable_count = compute_passing_lane_occlusion(bx, by, att_x, att_y, def_x, def_y)
        feat_passing_lane_occlusion[t] = occ_pct
        feat_viable_passing_options[t] = viable_count
        
        # Possession Duration Tracking
        if t == 0:
            feat_possession_duration_s[t] = 0.0
        else:
            if df["possession_team"].iloc[t] == df["possession_team"].iloc[t-1]:
                feat_possession_duration_s[t] = feat_possession_duration_s[t-1] + (1.0 / SAMPLING_RATE_HZ)
            else:
                feat_possession_duration_s[t] = 0.0

        # Pitch Location Context
        # Ball position (normalized so attacking direction is always positive X)
        feat_ball_x[t] = bx if defending_direction == 1.0 else (PITCH_LENGTH_METERS - bx)
        feat_ball_y[t] = by
        # Distance to opponent goal (x = PITCH_LENGTH_METERS, y = PITCH_WIDTH_METERS/2)
        feat_dist_to_goal[t] = np.sqrt((feat_ball_x[t] - PITCH_LENGTH_METERS)**2 + (by - PITCH_WIDTH_METERS/2.0)**2)
        # Distance to sideline
        feat_dist_to_sideline[t] = min(by, PITCH_WIDTH_METERS - by)

        # Ball kinematics
        feat_ball_speed[t] = ball_speed_arr[t]
        feat_ball_progression_x[t] = df.get("Ball_vx", pd.Series(np.zeros(n_frames))).values[t] * (-defending_direction)

    # Populate DataFrame
    df["feat_def_dist_1"] = feat_def_dist_1
    df["feat_def_dist_2"] = feat_def_dist_2
    df["feat_def_dist_3"] = feat_def_dist_3
    df["feat_def_dist_mean3"] = feat_def_dist_mean3

    df["feat_closing_speed_max"] = feat_closing_speed_max
    df["feat_closing_speed_mean3"] = feat_closing_speed_mean3
    df["feat_pressure_index"] = feat_pressure_index

    df["feat_def_density_5m"] = feat_def_density_5m
    df["feat_def_density_10m"] = feat_def_density_10m
    df["feat_att_density_10m"] = feat_att_density_10m
    df["feat_numerical_advantage_10m"] = feat_numerical_advantage_10m

    df["feat_def_hull_area"] = feat_def_hull_area
    df["feat_def_block_width"] = feat_def_block_width
    df["feat_def_block_depth"] = feat_def_block_depth
    df["feat_def_dispersion"] = feat_def_dispersion
    df["feat_def_centroid_dist_to_ball"] = feat_def_centroid_dist_to_ball
    df["feat_def_line_height"] = feat_def_line_height
    df["feat_ball_x"] = feat_ball_x
    df["feat_ball_y"] = feat_ball_y
    df["feat_dist_to_goal"] = feat_dist_to_goal
    df["feat_dist_to_sideline"] = feat_dist_to_sideline
    df["feat_passing_lane_occlusion"] = feat_passing_lane_occlusion
    df["feat_viable_passing_options"] = feat_viable_passing_options
    df["feat_possession_duration_s"] = feat_possession_duration_s
    df["feat_ball_speed"] = feat_ball_speed
    df["feat_ball_progression_x"] = feat_ball_progression_x

    # Strictly Backward Temporal Rolling Features (center=False, lookback: 1.5s = FEATURE_LOOKBACK_FRAMES)
    w = FEATURE_LOOKBACK_FRAMES
    df["feat_roll_pressure_index_mean"] = df["feat_pressure_index"].rolling(w, min_periods=1, center=False).mean()
    df["feat_roll_pressure_index_max"] = df["feat_pressure_index"].rolling(w, min_periods=1, center=False).max()
    df["feat_roll_closing_speed_mean"] = df["feat_closing_speed_mean3"].rolling(w, min_periods=1, center=False).mean()
    df["feat_roll_dist_mean3_min"] = df["feat_def_dist_mean3"].rolling(w, min_periods=1, center=False).min()
    df["feat_roll_hull_area_change"] = df["feat_def_hull_area"].diff(periods=w).fillna(0.0)

    return df


FEATURE_COLUMNS = [
    "feat_def_dist_1",
    "feat_def_dist_2",
    "feat_def_dist_3",
    "feat_def_dist_mean3",
    "feat_closing_speed_max",
    "feat_closing_speed_mean3",
    "feat_pressure_index",
    "feat_def_density_5m",
    "feat_def_density_10m",
    "feat_att_density_10m",
    "feat_numerical_advantage_10m",
    "feat_def_hull_area",
    "feat_def_block_width",
    "feat_def_block_depth",
    "feat_passing_lane_occlusion",
    "feat_def_dispersion",
    "feat_def_centroid_dist_to_ball",
    "feat_def_line_height",
    "feat_ball_speed",
    "feat_ball_progression_x",
    "feat_roll_pressure_index_mean",
    "feat_roll_pressure_index_max",
    "feat_roll_closing_speed_mean",
    "feat_roll_dist_mean3_min",
    "feat_roll_hull_area_change",
    "feat_ball_x",
    "feat_ball_y",
    "feat_dist_to_goal",
    "feat_dist_to_sideline",
    "feat_viable_passing_options",
    "feat_possession_duration_s"
]

FEATURE_DISPLAY_NAMES = {
    "feat_def_dist_1": "1st-Nearest Defender Dist (m)",
    "feat_def_dist_2": "2nd-Nearest Defender Dist (m)",
    "feat_def_dist_3": "3rd-Nearest Defender Dist (m)",
    "feat_def_dist_mean3": "Mean Distance of 3 Defenders (m)",
    "feat_closing_speed_max": "Max Defender Closing Speed (m/s)",
    "feat_closing_speed_mean3": "Mean 3-Defender Closing Speed (m/s)",
    "feat_pressure_index": "Normalized Pressure Index (0-1)",
    "feat_def_density_5m": "Defender Count within 5m",
    "feat_def_density_10m": "Defender Count within 10m",
    "feat_att_density_10m": "Attacker Count within 10m",
    "feat_numerical_advantage_10m": "Numerical Overload within 10m",
    "feat_def_hull_area": "Defensive Convex Hull Area (m²)",
    "feat_def_block_width": "Defensive Block Width (m)",
    "feat_def_block_depth": "Defensive Block Depth (m)",
    "feat_passing_lane_occlusion": "Passing Lane Occlusion (%)",
    "feat_def_dispersion": "Defensive Spatial Dispersion (m)",
    "feat_def_centroid_dist_to_ball": "Defensive Centroid Dist to Ball (m)",
    "feat_def_line_height": "Defensive Line Height (m)",
    "feat_ball_speed": "Ball Speed (m/s)",
    "feat_ball_progression_x": "Ball Progression Velocity (m/s)",
    "feat_roll_pressure_index_mean": "Rolling Mean Pressure Index (1.5s)",
    "feat_roll_pressure_index_max": "Rolling Peak Pressure Index (1.5s)",
    "feat_roll_closing_speed_mean": "Rolling Mean Closing Speed (1.5s)",
    "feat_roll_dist_mean3_min": "Rolling Min 3-Defender Dist (1.5s)",
    "feat_roll_hull_area_change": "Rate of Compactness Compression (m²/s)",
    "feat_ball_x": "Ball Normalized X (m)",
    "feat_ball_y": "Ball Y (m)",
    "feat_dist_to_goal": "Distance to Opponent Goal (m)",
    "feat_dist_to_sideline": "Distance to Sideline (m)",
    "feat_viable_passing_options": "Viable Passing Options",
    "feat_possession_duration_s": "Possession Duration (s)"
}
