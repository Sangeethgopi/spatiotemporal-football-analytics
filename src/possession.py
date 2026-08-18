"""
Possession Assignment & Tracking-Derived Turnover Derivation Module.
Implements a 3-State Hysteresis State Machine (Controlled Home, Controlled Away, Contested/Flight)
with strict temporal hold thresholds to prevent false micro-possession flips.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.config import (
    POSSESSION_BALL_PROXIMITY_M,
    POSSESSION_MIN_HOLD_FRAMES,
    TURNOVER_MIN_HOLD_FRAMES,
    TURNOVER_LOOKAHEAD_FRAMES,
    SAMPLING_RATE_HZ
)
from src.data_loader import get_player_columns


def assign_frame_possession(
    df: pd.DataFrame,
    dist_threshold: float = POSSESSION_BALL_PROXIMITY_M,
    min_hold_frames: int = POSSESSION_MIN_HOLD_FRAMES
) -> pd.DataFrame:
    """
    Computes Euclidean distance between the ball and every player per frame.
    Applies a 3-state hysteresis filter:
      - Controlled_Home: Ball within 2.0m of Home player
      - Controlled_Away: Ball within 2.0m of Away player
      - Contested / In-Flight: Ball > 2.0m from all players (passes, loose balls, shots)
    Sustained team possession requires continuous control for at least `min_hold_frames`.
    """
    df_out = df.copy()

    home_players = get_player_columns(df_out, "Home")
    away_players = get_player_columns(df_out, "Away")

    ball_x = df_out["Ball_x"].values
    ball_y = df_out["Ball_y"].values

    n_frames = len(df_out)
    nearest_player = np.array(["None"] * n_frames, dtype=object)
    nearest_team = np.array(["Contested"] * n_frames, dtype=object)
    min_distances = np.full(n_frames, np.inf)

    # Check all home players
    for p_name, (x_col, y_col) in home_players.items():
        px = df_out[x_col].values
        py = df_out[y_col].values
        dist = np.sqrt((px - ball_x)**2 + (py - ball_y)**2)
        dist[np.isnan(dist)] = np.inf

        closer = dist < min_distances
        min_distances[closer] = dist[closer]
        nearest_player[closer] = p_name
        nearest_team[closer] = "Home"

    # Check all away players
    for p_name, (x_col, y_col) in away_players.items():
        px = df_out[x_col].values
        py = df_out[y_col].values
        dist = np.sqrt((px - ball_x)**2 + (py - ball_y)**2)
        dist[np.isnan(dist)] = np.inf

        closer = dist < min_distances
        min_distances[closer] = dist[closer]
        nearest_player[closer] = p_name
        nearest_team[closer] = "Away"

    # Mark loose balls / passes where nearest player exceeds direct control radius (2.0m)
    loose_ball = min_distances > dist_threshold
    nearest_team[loose_ball] = "Contested"
    nearest_player[loose_ball] = "None"

    df_out["raw_possession_team"] = nearest_team
    df_out["raw_possession_player"] = nearest_player
    df_out["ball_carrier_distance"] = min_distances

    # Hysteresis Filter: A team retains tactical possession through passes (Contested states)
    # until the opposing team establishes controlled possession for >= min_hold_frames
    settled_team = []
    current_poss_team = "Home"
    consecutive_opp_frames = 0
    candidate_team = None

    for i in range(n_frames):
        raw_t = nearest_team[i]

        if raw_t in ("Home", "Away"):
            if raw_t == current_poss_team:
                consecutive_opp_frames = 0
                candidate_team = None
            else:
                if candidate_team == raw_t:
                    consecutive_opp_frames += 1
                else:
                    candidate_team = raw_t
                    consecutive_opp_frames = 1

                if consecutive_opp_frames >= min_hold_frames:
                    # Successful possession switch
                    current_poss_team = candidate_team
                    consecutive_opp_frames = 0
                    candidate_team = None
        else:
            # Ball in flight / loose: reset opponent takeover counter
            consecutive_opp_frames = 0
            candidate_team = None

        settled_team.append(current_poss_team)

    df_out["possession_team"] = settled_team
    return df_out


def derive_turnover_events(
    df_possession: pd.DataFrame,
    min_hold_frames: int = TURNOVER_MIN_HOLD_FRAMES,
    forced_turnover_threshold_m: float = 3.0
) -> pd.DataFrame:
    """
    Extracts validated tracking-derived turnover events where settled possession
    shifts from Team A -> Team B and is held for at least `min_hold_frames` (2.0s).
    Filters out unforced errors by requiring at least one pressing defender to be
    within `forced_turnover_threshold_m` at the moment possession is lost.
    """
    teams = df_possession["possession_team"].values
    n_frames = len(teams)

    turnover_rows = []
    current_team = None
    current_start_idx = 0

    home_players = get_player_columns(df_possession, "Home")
    away_players = get_player_columns(df_possession, "Away")

    for i in range(n_frames):
        t = teams[i]
        if current_team is None:
            current_team = t
            current_start_idx = i
        elif t != current_team:
            prev_duration_frames = i - current_start_idx
            prev_duration_sec = prev_duration_frames / SAMPLING_RATE_HZ

            # Valid turnover requires preceding team possessed ball for >= 3.0s (75 frames)
            if prev_duration_frames >= 75:
                # The moment of loss is approximately when the winning team first took control
                loss_idx = max(0, i - min_hold_frames)
                ball_x = df_possession.iloc[loss_idx]["Ball_x"]
                ball_y = df_possession.iloc[loss_idx]["Ball_y"]
                
                # Check defensive proximity at moment of loss
                pressing_players = home_players if t == "Home" else away_players
                min_def_dist = np.inf
                for _, (px_col, py_col) in pressing_players.items():
                    px = df_possession.iloc[loss_idx][px_col]
                    py = df_possession.iloc[loss_idx][py_col]
                    if not np.isnan(px) and not np.isnan(py):
                        dist = np.sqrt((px - ball_x)**2 + (py - ball_y)**2)
                        if dist < min_def_dist:
                            min_def_dist = dist
                
                # Only record if it was a forced turnover (defender nearby)
                if min_def_dist <= forced_turnover_threshold_m:
                    turnover_frame = df_possession.iloc[i]["Frame"]
                    turnover_time = df_possession.iloc[i].get("Time_s", turnover_frame / SAMPLING_RATE_HZ)

                    turnover_rows.append({
                        "turnover_id": len(turnover_rows) + 1,
                        "frame_idx": i,
                        "frame": turnover_frame,
                        "time_s": turnover_time,
                        "losing_team": current_team,
                        "winning_team": t,
                        "pressing_team": t,  # The team that forced the turnover
                        "turnover_x": ball_x,
                        "turnover_y": ball_y,
                        "prior_possession_duration_s": prev_duration_sec,
                        "defender_proximity_m": min_def_dist
                    })

            current_team = t
            current_start_idx = i

    return pd.DataFrame(turnover_rows)


def label_turnover_horizons(
    df: pd.DataFrame,
    turnovers_df: pd.DataFrame,
    horizon_frames: int = TURNOVER_LOOKAHEAD_FRAMES
) -> pd.DataFrame:
    """
    Creates weak binary supervision labels:
    `target_press_trigger` = 1 if defending team forces a turnover within horizon_frames (4.0s), else 0.
    Also computes multi-class tactical outcomes based on ball progression over the horizon.
    """
    df_labeled = df.copy()
    n_frames = len(df_labeled)

    target_home_press = np.zeros(n_frames, dtype=int)
    target_away_press = np.zeros(n_frames, dtype=int)

    if not turnovers_df.empty:
        for _, to_event in turnovers_df.iterrows():
            to_idx = int(to_event["frame_idx"])
            winning_team = to_event["winning_team"]

            start_idx = max(0, to_idx - horizon_frames)
            if winning_team == "Home":
                target_home_press[start_idx:to_idx] = 1
            elif winning_team == "Away":
                target_away_press[start_idx:to_idx] = 1

    df_labeled["target_home_press_success"] = target_home_press
    df_labeled["target_away_press_success"] = target_away_press

    # Unified target: is the defending team currently in an active pressing window that forces a turnover?
    poss_arr = df_labeled["possession_team"].values
    defending_turnover_target = np.where(
        poss_arr == "Home",
        target_away_press,
        np.where(poss_arr == "Away", target_home_press, 0)
    )

    df_labeled["target_press_trigger"] = defending_turnover_target
    
    # --- Nuanced Multi-Class Tactical Outcomes ---
    # We evaluate the ball's spatial trajectory N seconds into the future.
    # 0 = Neutral/Unknown
    # 1 = Ball Recovery (Turnover)
    # 2 = Forced Retreat (Retained, but moved backwards > 10m)
    # 3 = Forced Lateral (Retained, but < 5m forward progress)
    # 4 = Press Beaten (Retained, > 10m forward progress)
    # 5 = Dangerous Transition (Retained, > 20m forward progress)
    
    outcomes = np.zeros(n_frames, dtype=int)
    ball_x = df_labeled["Ball_x"].values
    
    for i in range(n_frames - horizon_frames):
        if poss_arr[i] not in ("Home", "Away"):
            continue
            
        defending_team = "Away" if poss_arr[i] == "Home" else "Home"
        future_idx = i + horizon_frames
        
        # Did a turnover happen between i and future_idx?
        if poss_arr[future_idx] == defending_team:
            outcomes[i] = 1 # Ball Recovery
        else:
            # Ball retained by the attacking team
            x_now = ball_x[i]
            x_future = ball_x[future_idx]
            
            # Forward direction for Home is positive X (0 -> 105), Away is negative X (105 -> 0)
            if poss_arr[i] == "Home":
                progress = x_future - x_now
            else:
                progress = x_now - x_future
                
            if progress < -10.0:
                outcomes[i] = 2 # Forced Retreat
            elif progress < 5.0:
                outcomes[i] = 3 # Forced Lateral
            elif progress < 20.0:
                outcomes[i] = 4 # Press Beaten
            else:
                outcomes[i] = 5 # Dangerous Transition
                
    df_labeled["tactical_outcome"] = outcomes

    return df_labeled

