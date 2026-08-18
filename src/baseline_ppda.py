"""
PPDA & Spatial Heuristic Baseline Module.
Defines Macro-level PPDA metrics and the Spatial Proximity Heuristic Baseline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from src.config import PITCH_LENGTH_METERS, ATTACKING_60_PCT_THRESHOLD, SAMPLING_RATE_HZ


def compute_match_ppda(df_features: pd.DataFrame, turnovers_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes traditional match-level PPDA (Passes Allowed Per Defensive Action)
    for Home and Away teams in the opponent's defensive 60% of the pitch.
    """
    home_ppda_zone_x = PITCH_LENGTH_METERS - ATTACKING_60_PCT_THRESHOLD  # 63m
    away_ppda_zone_x = ATTACKING_60_PCT_THRESHOLD                         # 42m

    # Approximate opponent passes as possession time / 2.5s avg pass duration
    home_pressing_frames = df_features[
        (df_features["possession_team"] == "Away") &
        (df_features["Ball_x"] <= home_ppda_zone_x)
    ]
    away_pressing_frames = df_features[
        (df_features["possession_team"] == "Home") &
        (df_features["Ball_x"] >= away_ppda_zone_x)
    ]

    home_opp_passes_est = len(home_pressing_frames) / (SAMPLING_RATE_HZ * 2.5)
    away_opp_passes_est = len(away_pressing_frames) / (SAMPLING_RATE_HZ * 2.5)

    if not turnovers_df.empty:
        home_def_actions = len(turnovers_df[
            (turnovers_df["winning_team"] == "Home") &
            (turnovers_df["turnover_x"] <= home_ppda_zone_x)
        ])
        away_def_actions = len(turnovers_df[
            (turnovers_df["winning_team"] == "Away") &
            (turnovers_df["turnover_x"] >= away_ppda_zone_x)
        ])
    else:
        home_def_actions = 1
        away_def_actions = 1

    home_def_actions = max(1, home_def_actions)
    away_def_actions = max(1, away_def_actions)

    return {
        "home_ppda": float(home_opp_passes_est / home_def_actions),
        "away_ppda": float(away_opp_passes_est / away_def_actions),
        "home_defensive_actions": home_def_actions,
        "away_defensive_actions": away_def_actions
    }


class RollingPPDABaselineClassifier:
    """
    Rolling PPDA Baseline:
    Predicts a press trigger when the rolling PPDA (Passes allowed per Defensive Action)
    drops below a tactical threshold. Since we operate on pure tracking data, we use
    time-in-possession and close-proximity durations as proxies for passes and actions.
    """
    def __init__(self, window_sec: float = 15.0):
        self.window_frames = int(window_sec * SAMPLING_RATE_HZ)

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dist = X.get("feat_def_dist_1", pd.Series(np.full(len(X), 25.0))).fillna(25.0).values
        
        # Proxy for defensive actions: frames where nearest defender is within 2.5m
        def_action_frames = (dist < 2.5).astype(float)
        s = pd.Series(def_action_frames)
        rolling_def_actions = s.rolling(self.window_frames, min_periods=1).sum().values
        def_action_sec = rolling_def_actions / SAMPLING_RATE_HZ
        
        # Proxy for passes allowed: time elapsed divided by avg pass duration (3.0s)
        # Using continuous frame index as a proxy for time in out-of-possession segments
        rolling_passes = np.minimum(np.arange(1, len(X) + 1), self.window_frames) / (3.0 * SAMPLING_RATE_HZ)
        
        # Calculate Rolling PPDA
        ppda = rolling_passes / np.maximum(0.25, def_action_sec)
        
        # Scale PPDA to probability: Lower PPDA means higher pressing intensity
        # Typical PPDA: < 8 is intense pressing, > 15 is passive.
        score = np.clip(1.0 - (ppda / 15.0), 0.0, 1.0)
        
        return np.column_stack([1.0 - score, score])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class DistanceBaselineClassifier:
    """
    Baseline 1: Distance rule.
    Predicts a press trigger if the nearest defender is within X meters.
    """
    def __init__(self, dist_threshold: float = 2.5):
        self.dist_threshold = dist_threshold

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dist = X.get("feat_def_dist_1", pd.Series(np.full(len(X), 25.0))).fillna(25.0).values
        # Convert distance to a probability score: closer = higher prob
        score = np.clip(1.0 - (dist / (self.dist_threshold * 2)), 0.0, 1.0)
        # Binarize strict threshold for true rule-based baseline
        score = np.where(dist < self.dist_threshold, np.maximum(score, 0.6), np.minimum(score, 0.4))
        return np.column_stack([1.0 - score, score])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class VelocityDistanceBaselineClassifier:
    """
    Baseline 2: Distance + Velocity rule.
    Predicts a press trigger if the nearest defender is within X meters AND closing faster than Y m/s.
    """
    def __init__(self, dist_threshold: float = 2.5, vel_threshold: float = 3.0):
        self.dist_threshold = dist_threshold
        self.vel_threshold = vel_threshold

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dist = X.get("feat_def_dist_1", pd.Series(np.full(len(X), 25.0))).fillna(25.0).values
        closing_speed = X.get("feat_closing_speed_max", pd.Series(np.zeros(len(X)))).fillna(0.0).values
        
        # Rule condition
        rule_met = (dist < self.dist_threshold) & (closing_speed > self.vel_threshold)
        
        # Smooth scoring for ROC curve
        dist_score = np.clip(1.0 - (dist / (self.dist_threshold * 2)), 0.0, 1.0)
        vel_score = np.clip(closing_speed / (self.vel_threshold * 2), 0.0, 1.0)
        base_score = (dist_score + vel_score) / 2.0
        
        score = np.where(rule_met, np.maximum(base_score, 0.6), np.minimum(base_score, 0.4))
        return np.column_stack([1.0 - score, score])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

