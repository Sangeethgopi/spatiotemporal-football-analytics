"""
Main End-to-End Pipeline Orchestrator.
Connects data ingestion, possession assignment, causal feature engineering,
calibrated model training, and lead-time evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from src.baseline_ppda import compute_match_ppda, RollingPPDABaselineClassifier
from src.model import PressTriggerClassifier
from src.evaluate import (
    evaluate_predictions,
    compare_model_vs_baseline,
    compute_threshold_sweep,
    compute_lead_time_distribution,
    compute_time_resolved_intensity,
    extract_top_press_triggers
)


def run_pipeline(
    match_id: str = "match_1",
    data_dir: str = "data",
    model_type: str = "hist_gb",
    train_val_split_ratio: float = 0.70,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Executes full sports analytics pressing pipeline end-to-end.
    """
    print(f"=== [1/6] Ingesting Tracking Data for {match_id} ===")
    paths = ensure_dataset_available(data_dir=data_dir, match_id=match_id)
    df_raw = load_match_tracking(paths["home"], paths["away"])
    duration_s = len(df_raw) / 25.0
    print(f"Loaded {len(df_raw)} frames ({duration_s:.1f} seconds / {duration_s/60.0:.1f} minutes).")

    print("=== [2/6] Assigning Hysteresis Possession & Deriving Turnovers ===")
    df_poss = assign_frame_possession(df_raw)
    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    print(f"Identified {len(turnovers_df)} tracking-derived possession turnovers.")

    print("=== [3/6] Engineering Causal Spatial-Temporal Pressing Features ===")
    df_features = extract_pressing_features(df_labeled)

    # Sanity Check Physical Kinematics
    max_c_spd = df_features["feat_closing_speed_max"].max()
    print(f"Kinematic Sanity Check: Max Closing Speed = {max_c_spd:.2f} m/s (Physical limit <= 10.5 m/s: {'PASS' if max_c_spd <= 10.5 else 'FAIL'})")

    # Macro PPDA Metrics
    ppda_stats = compute_match_ppda(df_features, turnovers_df)
    print(f"Match PPDA -> Home: {ppda_stats['home_ppda']:.2f}, Away: {ppda_stats['away_ppda']:.2f}")

    print("=== [4/6] Training Calibrated ML Model & Spatial Baseline ===")
    n_frames = len(df_features)
    split_idx = int(n_frames * train_val_split_ratio)

    df_train = df_features.iloc[:split_idx]
    df_val = df_features.iloc[split_idx:]

    y_train = df_train["target_press_trigger"].values
    y_val = df_val["target_press_trigger"].values

    # Train Calibrated ML Classifier
    clf = PressTriggerClassifier(model_type=model_type, random_state=42)
    clf.fit(df_train, y_train)

    # Fit Baseline
    baseline_clf = RollingPPDABaselineClassifier(window_sec=15.0)
    baseline_clf.fit(df_train, y_train)

    print("=== [5/6] Evaluating Models on Temporal Holdout Split ===")
    model_val_probs = clf.predict_proba(df_val)[:, 1]
    baseline_val_probs = baseline_clf.predict_proba(df_val)[:, 1]

    comparison_df = compare_model_vs_baseline(y_val, model_val_probs, baseline_val_probs, threshold=0.50)
    print("\n" + comparison_df.to_string(index=False) + "\n")

    # Feature Importance with 95% CI
    feat_imp = clf.get_feature_importances(df_val, y_val)
    print("Top Predictive Press Features (Permutation Importance on ROC-AUC):")
    print(feat_imp.head(5)[["feature", "importance_mean", "importance_std"]].to_string(index=False))

    print("=== [6/6] Generating Full Match Inference & Lead-Time Analysis ===")
    full_probs = clf.predict_proba(df_features)[:, 1]
    df_features["press_trigger_prob"] = full_probs

    top_triggers = extract_top_press_triggers(df_features, top_n=10, threshold=0.50)
    lead_time_stats = compute_lead_time_distribution(df_features, turnovers_df, threshold=0.50)
    timeline_profile = compute_time_resolved_intensity(df_features)
    threshold_sweep_df = compute_threshold_sweep(y_val, model_val_probs)

    print(f"Empirical Lead Time Stats: Median = {lead_time_stats['median_lead_time']:.2f}s, IQR = [{lead_time_stats['iqr_lower']:.2f}s - {lead_time_stats['iqr_upper']:.2f}s]")

    results = {
        "match_id": match_id,
        "dataframe": df_features,
        "turnovers": turnovers_df,
        "ppda_stats": ppda_stats,
        "comparison_metrics": comparison_df,
        "feature_importances": feat_imp,
        "top_triggers": top_triggers,
        "lead_time_stats": lead_time_stats,
        "timeline_profile": timeline_profile,
        "threshold_sweep": threshold_sweep_df,
        "model": clf,
        "baseline": baseline_clf
    }

    if save_artifacts:
        out_dir = os.path.join(data_dir, "results")
        os.makedirs(out_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(out_dir, f"{match_id}_comparison_metrics.csv"), index=False)
        feat_imp.to_csv(os.path.join(out_dir, f"{match_id}_feature_importance.csv"), index=False)
        turnovers_df.to_csv(os.path.join(out_dir, f"{match_id}_turnovers.csv"), index=False)
        threshold_sweep_df.to_csv(os.path.join(out_dir, f"{match_id}_threshold_sweep.csv"), index=False)
        print(f"Results saved to: {out_dir}")

    return results


if __name__ == "__main__":
    results = run_pipeline("match_1", data_dir="data")
