"""
Data loader for Metrica Sports tracking data.
Parses CSV headers, standardizes coordinate systems, and converts to metric dimensions.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from src.config import (
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
    SAMPLING_RATE_HZ,
    MAX_REALISTIC_PLAYER_SPEED_M_S,
    MAX_REALISTIC_BALL_SPEED_M_S
)
import kloppy


def load_metrica_csv(csv_path: str, team_prefix: str = "Home", max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Parses a Metrica Sports tracking CSV file.
    Metrica CSV format has 3 header rows:
      Row 0: Team metadata (e.g. ,,,Home,,Home,...)
      Row 1: Player Numbers (e.g. ,,,11,,1,,2,...)
      Row 2: Labels (e.g. Period,Frame,Time [s],Player11,,Player1,,...Ball,)
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        raise FileNotFoundError(f"Tracking file missing or empty: {csv_path}")

    # Read the header lines
    with open(csv_path, 'r', encoding='utf-8') as f:
        line_0 = [c.strip() for c in f.readline().split(',')]
        line_1 = [c.strip() for c in f.readline().split(',')]
        line_2 = [c.strip() for c in f.readline().split(',')]

    n_cols = max(len(line_0), len(line_1), len(line_2))
    # Pad lists to same length
    line_0 += [''] * (n_cols - len(line_0))
    line_1 += [''] * (n_cols - len(line_1))
    line_2 += [''] * (n_cols - len(line_2))

    # If line 2 already has formatted column names (e.g. Home_1_x, Ball_x, etc.)
    column_names = []
    seen = {}

    for idx in range(len(line_2)):
        l0 = line_0[idx] if idx < len(line_0) else ""
        l1 = line_1[idx] if idx < len(line_1) else ""
        l2 = line_2[idx] if idx < len(line_2) else ""

        if idx == 0 or l2 == "Period":
            col_name = "Period"
        elif idx == 1 or l2 == "Frame":
            col_name = "Frame"
        elif idx == 2 or "Time" in l2:
            col_name = "Time_s"
        elif l2.endswith("_x") or l2.endswith("_y"):
            col_name = l2
        elif "Ball" in l2 or "Ball" in l1 or "Ball" in l0:
            if idx > 0 and column_names[-1] == "Ball_x":
                col_name = "Ball_y"
            else:
                col_name = "Ball_x"
        elif l1 or l2:
            p_num = re.search(r'\d+', l1 or l2)
            num = p_num.group(0) if p_num else str(idx)
            # If previous column was Home_N_x, this one is Home_N_y
            if idx > 0 and column_names[-1] == f"{team_prefix}_{num}_x":
                col_name = f"{team_prefix}_{num}_y"
            else:
                col_name = f"{team_prefix}_{num}_x"
        else:
            col_name = f"{team_prefix}_col{idx}"

        # Ensure uniqueness
        if col_name in seen:
            seen[col_name] += 1
            col_name = f"{col_name}_{seen[col_name]}"
        else:
            seen[col_name] = 1

        column_names.append(col_name)

    # Load data skipping headers
    df = pd.read_csv(
        csv_path,
        skiprows=3,
        header=None,
        names=column_names,
        nrows=max_rows,
        low_memory=False
    )

    # Convert numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def to_metric_coordinates(
    df: pd.DataFrame,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS
) -> pd.DataFrame:
    """
    Transforms normalized [0, 1] Metrica coordinates to meters [0, 105] x [0, 68].
    Handles pitch origin at bottom-left (0,0) to top-right (105, 68).
    """
    df_metric = df.copy()
    for col in df_metric.columns:
        if col.endswith('_x') or col == 'Ball_x':
            # Check if values appear normalized (mean <= 1.5)
            valid_vals = df_metric[col].dropna()
            if len(valid_vals) > 0 and valid_vals.max() <= 1.5:
                df_metric[col] = df_metric[col] * pitch_length
        elif col.endswith('_y') or col == 'Ball_y':
            valid_vals = df_metric[col].dropna()
            if len(valid_vals) > 0 and valid_vals.max() <= 1.5:
                # Metrica y is often inverted (0 is top, 1 is bottom); normalize to standard cartesian
                df_metric[col] = df_metric[col] * pitch_width
    return df_metric


def compute_velocities(
    df: pd.DataFrame,
    sampling_rate: int = SAMPLING_RATE_HZ,
    max_player_speed: float = MAX_REALISTIC_PLAYER_SPEED_M_S,
    max_ball_speed: float = MAX_REALISTIC_BALL_SPEED_M_S
) -> pd.DataFrame:
    """
    Computes velocity vectors (vx, vy) and speed (m/s) for all players and ball.
    STRICTLY CAUSAL: Uses backward finite differences and non-centered smoothing
    to completely prevent future data leakage (t uses only data <= t).
    Enforces strict physical maximum speed bounds (players <= 10.5 m/s, ball <= 36.0 m/s).
    """
    df_vel = df.copy()
    dt = 1.0 / sampling_rate

    # Identify all entities
    x_cols = [c for c in df_vel.columns if c.endswith('_x')]
    for x_col in x_cols:
        prefix = x_col[:-2]
        y_col = f"{prefix}_y"
        if y_col in df_vel.columns:
            # Backward finite differences (strictly causal)
            vx_raw = df_vel[x_col].diff() / dt
            vy_raw = df_vel[y_col].diff() / dt

            # Strictly causal 5-frame backward rolling mean (center=False)
            vx_smooth = vx_raw.rolling(window=5, min_periods=1, center=False).mean().fillna(0.0)
            vy_smooth = vy_raw.rolling(window=5, min_periods=1, center=False).mean().fillna(0.0)
            speed_raw = np.sqrt(vx_smooth**2 + vy_smooth**2)

            # Strict physical speed clipping
            is_ball = ("ball" in prefix.lower())
            speed_cap = max_ball_speed if is_ball else max_player_speed

            # Scale velocity vector proportionally if speed exceeds physical maximum
            overspeed_mask = speed_raw > speed_cap
            scale_factors = np.ones_like(speed_raw)
            scale_factors[overspeed_mask] = speed_cap / np.maximum(speed_raw[overspeed_mask], 1e-6)

            vx_clamped = vx_smooth * scale_factors
            vy_clamped = vy_smooth * scale_factors
            speed_clamped = np.clip(speed_raw, 0.0, speed_cap)

            df_vel[f"{prefix}_vx"] = vx_clamped
            df_vel[f"{prefix}_vy"] = vy_clamped
            df_vel[f"{prefix}_speed"] = speed_clamped

    return df_vel


def load_match_tracking(
    home_csv: str,
    away_csv: str,
    pitch_length: float = PITCH_LENGTH_METERS,
    pitch_width: float = PITCH_WIDTH_METERS
) -> pd.DataFrame:
    """
    Loads both Home and Away tracking files, converts to metric space,
    computes velocities, and merges into a unified DataFrame per frame.
    """
    df_home = load_metrica_csv(home_csv, team_prefix="Home")
    df_away = load_metrica_csv(away_csv, team_prefix="Away")

    # Metric coordinates
    df_home_m = to_metric_coordinates(df_home, pitch_length, pitch_width)
    df_away_m = to_metric_coordinates(df_away, pitch_length, pitch_width)

    # If away CSV contains duplicate Ball columns, drop them from away
    away_ball_cols = [c for c in df_away_m.columns if "Ball" in c]
    if away_ball_cols and "Ball_x" in df_home_m.columns:
        df_away_m = df_away_m.drop(columns=away_ball_cols)

    # Merge on Frame and Period
    merge_cols = ["Period", "Frame"]
    if "Time_s" in df_home_m.columns and "Time_s" in df_away_m.columns:
        df_away_m = df_away_m.drop(columns=["Time_s"])

    df_merged = pd.merge(df_home_m, df_away_m, on=merge_cols, how="inner")
    df_merged = compute_velocities(df_merged)

    return df_merged


def load_match_tracking_epts(metadata_xml: str, raw_txt: str) -> pd.DataFrame:
    """
    Ingests FIFA EPTS format tracking data (e.g., Metrica Match 3) using kloppy.
    Returns standard dataframe matched to our format.
    """
    import kloppy
    dataset = kloppy.epts.load(
        meta_data=metadata_xml,
        raw_data=raw_txt,
        coordinates="metrica",
        length=PITCH_LENGTH_METERS,
        width=PITCH_WIDTH_METERS
    )
    df = dataset.to_df()
    
    # Map kloppy dataframe output to our expected format
    rename_map = {}
    for col in df.columns:
        if "home_" in col:
            rename_map[col] = col.replace("home_", "Home_").replace("_x", "_x").replace("_y", "_y")
        elif "away_" in col:
            rename_map[col] = col.replace("away_", "Away_").replace("_x", "_x").replace("_y", "_y")
        elif col == "ball_x":
            rename_map[col] = "Ball_x"
        elif col == "ball_y":
            rename_map[col] = "Ball_y"
        elif col == "timestamp":
            rename_map[col] = "Time_s"
        elif col == "frame_id":
            rename_map[col] = "Frame"
            
    df = df.rename(columns=rename_map)
    df["Period"] = 1 # Simplified, we can infer period from frame jumps if needed
    
    df = compute_velocities(df)
    return df


def load_match_tracking_skillcorner(json_path: str) -> pd.DataFrame:
    """
    Ingests SkillCorner broadcast tracking data using kloppy for out-of-domain evaluation.
    """
    import kloppy
    dataset = kloppy.skillcorner.load(
        meta_data=json_path.replace("tracking", "metadata"),
        raw_data=json_path,
        coordinates="skillcorner",
        length=PITCH_LENGTH_METERS,
        width=PITCH_WIDTH_METERS
    )
    df = dataset.to_df()
    # Similar mapping logic would apply here as in EPTS
    return df


def get_player_columns(df: pd.DataFrame, team: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns a dictionary of {player_id: (x_col, y_col)} for active players in the team.
    """
    players = {}
    pattern = re.compile(rf"^{team}_(\d+)_x$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            pid = match.group(1)
            y_col = f"{team}_{pid}_y"
            if y_col in df.columns:
                players[f"{team}_{pid}"] = (col, y_col)
    return players
