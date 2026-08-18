"""
Unit and Integration Tests for Pressing Intensity & Trigger Detection System.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import PITCH_LENGTH_METERS, PITCH_WIDTH_METERS, MAX_REALISTIC_PLAYER_SPEED_M_S
from src.download_metrica import generate_kinematic_match_csv
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, compute_convex_hull_area, FEATURE_COLUMNS
from src.baseline_ppda import compute_match_ppda, SpatialProximityBaselineClassifier
from src.model import PressTriggerClassifier
from src.evaluate import evaluate_predictions, compute_lead_time_distribution


class TestPressingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(root_dir, "tests", "test_data")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.home_csv = os.path.join(cls.test_dir, "test_home.csv")
        cls.away_csv = os.path.join(cls.test_dir, "test_away.csv")
        generate_kinematic_match_csv(cls.home_csv, cls.away_csv, n_frames=1500, seed=123)

    def test_01_kinematic_data_loader(self):
        """Verify data loader loads CSV, converts coordinates, and bounds velocities."""
        df_raw = load_match_tracking(self.home_csv, self.away_csv)
        self.assertFalse(df_raw.empty)
        self.assertEqual(len(df_raw), 1500)
        self.assertIn("Ball_x", df_raw.columns)
        self.assertIn("Home_1_vx", df_raw.columns)

        # Bounds check for physical velocities (<= 10.5 m/s)
        h1_speed = df_raw["Home_1_speed"].max()
        self.assertLessEqual(h1_speed, MAX_REALISTIC_PLAYER_SPEED_M_S + 0.1)

    def test_02_possession_and_turnovers(self):
        """Verify 3-state possession hysteresis and turnover derivation."""
        df_raw = load_match_tracking(self.home_csv, self.away_csv)
        df_poss = assign_frame_possession(df_raw)
        self.assertIn("possession_team", df_poss.columns)

        turnovers_df = derive_turnover_events(df_poss)
        self.assertFalse(turnovers_df.empty)
        self.assertIn("winning_team", turnovers_df.columns)

        df_labeled = label_turnover_horizons(df_poss, turnovers_df)
        self.assertIn("target_press_trigger", df_labeled.columns)

    def test_03_causal_feature_extraction(self):
        """Verify feature engineering produces physically bounded, causal features."""
        df_raw = load_match_tracking(self.home_csv, self.away_csv)
        df_poss = assign_frame_possession(df_raw)
        turnovers_df = derive_turnover_events(df_poss)
        df_labeled = label_turnover_horizons(df_poss, turnovers_df)
        df_feat = extract_pressing_features(df_labeled)

        for col in FEATURE_COLUMNS:
            self.assertIn(col, df_feat.columns)

        # Verify closing speed bound
        max_closing = df_feat["feat_closing_speed_max"].max()
        self.assertLessEqual(max_closing, MAX_REALISTIC_PLAYER_SPEED_M_S + 0.1)

        # Verify pressure index is in [0, 1]
        self.assertGreaterEqual(df_feat["feat_pressure_index"].min(), 0.0)
        self.assertLessEqual(df_feat["feat_pressure_index"].max(), 1.0)

    def test_04_spatial_baseline(self):
        """Verify PPDA metrics and Spatial Baseline Classifier."""
        df_raw = load_match_tracking(self.home_csv, self.away_csv)
        df_poss = assign_frame_possession(df_raw)
        turnovers_df = derive_turnover_events(df_poss)
        df_labeled = label_turnover_horizons(df_poss, turnovers_df)
        df_feat = extract_pressing_features(df_labeled)

        ppda_stats = compute_match_ppda(df_feat, turnovers_df)
        self.assertIn("home_ppda", ppda_stats)
        self.assertGreater(ppda_stats["home_ppda"], 0)

        base_clf = SpatialProximityBaselineClassifier(proximity_threshold_m=2.5)
        preds = base_clf.predict(df_feat)
        self.assertEqual(len(preds), len(df_feat))

    def test_05_calibrated_model_and_lead_time(self):
        """Verify model calibration and empirical lead-time calculation."""
        df_raw = load_match_tracking(self.home_csv, self.away_csv)
        df_poss = assign_frame_possession(df_raw)
        turnovers_df = derive_turnover_events(df_poss)
        df_labeled = label_turnover_horizons(df_poss, turnovers_df)
        df_feat = extract_pressing_features(df_labeled)

        y = df_feat["target_press_trigger"].values
        clf = PressTriggerClassifier(model_type="hist_gb", random_state=42)
        clf.fit(df_feat, y)

        probs = clf.predict_proba(df_feat)[:, 1]
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        df_feat["press_trigger_prob"] = probs
        lead_stats = compute_lead_time_distribution(df_feat, turnovers_df)
        self.assertIn("median_lead_time", lead_stats)


if __name__ == "__main__":
    unittest.main()
