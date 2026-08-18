"""
Evaluation, Metrics, and Lead-Time Statistical Analysis Module.
Provides threshold sweep curves, empirical lead-time distributions, and baseline benchmarking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from src.config import SAMPLING_RATE_HZ


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes classification performance metrics at a specific decision threshold.
    """
    valid_mask = ~np.isnan(y_prob) & ~np.isnan(y_true)
    y_t = y_true[valid_mask].astype(int)
    y_p = y_prob[valid_mask]
    y_pred = (y_p >= threshold).astype(int)

    if len(np.unique(y_t)) < 2:
        roc_auc = 0.5
        pr_auc = 0.0
    else:
        roc_auc = float(roc_auc_score(y_t, y_p))
        pr_auc = float(average_precision_score(y_t, y_p))

    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    fpr, tpr, _ = roc_curve(y_t, y_p)
    precs, recs, _ = precision_recall_curve(y_t, y_p)
    brier = float(brier_score_loss(y_t, y_p))
    prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=10)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precs.tolist(), "recall": recs.tolist()},
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
    }


def compute_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generates precision, recall, and F1 trade-offs across decision thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 17)

    sweep_rows = []
    for thresh in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=thresh)
        sweep_rows.append({
            "threshold": round(float(thresh), 2),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1"],
            "true_positives": m["confusion_matrix"]["tp"],
            "false_positives": m["confusion_matrix"]["fp"],
            "false_negatives": m["confusion_matrix"]["fn"]
        })

    return pd.DataFrame(sweep_rows)


def compute_lead_time_distribution(
    df: pd.DataFrame,
    turnovers_df: pd.DataFrame,
    prob_col: str = "press_trigger_prob",
    threshold: float = 0.50,
    max_lookahead_sec: float = 5.0,
    n_bootstraps: int = 1000
) -> Dict[str, Any]:
    """
    Computes empirical lead-time (seconds prior to turnover when press trigger was first detected).
    Includes 95% Bootstrap Confidence Intervals.
    """
    if turnovers_df.empty or prob_col not in df.columns:
        return {
            "lead_times": [],
            "median_lead_time": 0.0,
            "mean_lead_time": 0.0,
            "iqr_lower": 0.0,
            "iqr_upper": 0.0,
            "detection_rate": 0.0
        }

    max_frames = int(max_lookahead_sec * SAMPLING_RATE_HZ)
    lead_times = []
    detected_count = 0

    for _, to_row in turnovers_df.iterrows():
        to_idx = int(to_row["frame_idx"])
        window_start = max(0, to_idx - max_frames)
        sub_probs = df.iloc[window_start:to_idx][prob_col].values

        # Find first index in window exceeding threshold
        trig_indices = np.where(sub_probs >= threshold)[0]
        if len(trig_indices) > 0:
            first_trig_offset = trig_indices[0]  # Offset from window_start
            first_trig_frame = window_start + first_trig_offset
            lead_time_sec = (to_idx - first_trig_frame) / SAMPLING_RATE_HZ
            lead_times.append(float(lead_time_sec))
            detected_count += 1

    if lead_times:
        arr = np.array(lead_times)
        q25, q75 = np.percentile(arr, [25, 75])
        
        # Bootstrap 95% CI for the mean lead time
        if n_bootstraps > 0 and len(arr) > 1:
            boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstraps)]
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_lower, ci_upper = float(np.mean(arr)), float(np.mean(arr))
            
        return {
            "lead_times": lead_times,
            "median_lead_time": float(np.median(arr)),
            "mean_lead_time": float(np.mean(arr)),
            "std_lead_time": float(np.std(arr)),
            "iqr_lower": float(q25),
            "iqr_upper": float(q75),
            "ci_lower_95": float(ci_lower),
            "ci_upper_95": float(ci_upper),
            "detection_rate": float(detected_count / len(turnovers_df))
        }
    else:
        return {
            "lead_times": [],
            "median_lead_time": 0.0,
            "mean_lead_time": 0.0,
            "std_lead_time": 0.0,
            "iqr_lower": 0.0,
            "iqr_upper": 0.0,
            "ci_lower_95": 0.0,
            "ci_upper_95": 0.0,
            "detection_rate": 0.0
        }


def compare_model_vs_baseline(
    y_true: np.ndarray,
    model_probs: np.ndarray,
    baseline_probs: np.ndarray,
    threshold: float = 0.5,
    all_baselines: Optional[Dict[str, np.ndarray]] = None
) -> pd.DataFrame:
    """
    Generates formal performance comparison table between ML Press Model and Baseline models.
    """
    rows = []
    
    if all_baselines is None:
        all_baselines = {"Baseline (Rolling PPDA)": baseline_probs}
        
    for name, probs in all_baselines.items():
        res_base = evaluate_predictions(y_true, probs, threshold=threshold)
        rows.append({
            "Methodology": name,
            "ROC-AUC": f"{res_base['roc_auc']:.3f}",
            "PR-AUC": f"{res_base['pr_auc']:.3f}",
            "Brier Score": f"{res_base['brier_score']:.3f}",
            "Precision": f"{res_base['precision']:.3f}",
            "Recall": f"{res_base['recall']:.3f}",
            "F1-Score": f"{res_base['f1']:.3f}",
            "True Positives": res_base["confusion_matrix"]["tp"],
            "False Positives": res_base["confusion_matrix"]["fp"]
        })

    res_model = evaluate_predictions(y_true, model_probs, threshold=threshold)
    rows.append({
        "Methodology": "Press Trigger Model (Ours)",
        "ROC-AUC": f"{res_model['roc_auc']:.3f}",
        "PR-AUC": f"{res_model['pr_auc']:.3f}",
        "Brier Score": f"{res_model['brier_score']:.3f}",
        "Precision": f"{res_model['precision']:.3f}",
        "Recall": f"{res_model['recall']:.3f}",
        "F1-Score": f"{res_model['f1']:.3f}",
        "True Positives": res_model["confusion_matrix"]["tp"],
        "False Positives": res_model["confusion_matrix"]["fp"]
    })

    return pd.DataFrame(rows)


def compute_time_resolved_intensity(
    df: pd.DataFrame,
    intensity_col: str = "press_trigger_prob",
    window_minutes: float = 15.0  # Standard 15-minute match intervals (0-15', 15-30', etc.)
) -> pd.DataFrame:
    """
    Computes time-resolved pressing intensity profiles.
    """
    df_temp = df.copy()
    if "Time_s" not in df_temp.columns:
        df_temp["Time_s"] = df_temp["Frame"] / SAMPLING_RATE_HZ

    df_temp["match_minute"] = df_temp["Time_s"] / 60.0
    interval_size = window_minutes
    df_temp["interval_bin"] = (df_temp["match_minute"] // interval_size) * interval_size
    df_temp["interval_label"] = df_temp["interval_bin"].apply(
        lambda b: f"{int(b):02d}'-{int(b + interval_size):02d}'"
    )

    profile = df_temp.groupby(["interval_label", "possession_team"])[intensity_col].agg(
        ["mean", "max", "count"]
    ).reset_index()

    return profile


def extract_top_press_triggers(
    df: pd.DataFrame,
    top_n: int = 10,
    prob_col: str = "press_trigger_prob",
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Extracts the highest-urgency press trigger moments across the match.
    """
    if prob_col not in df.columns:
        return pd.DataFrame()

    df_peaks = df[df[prob_col] >= threshold].copy()
    if df_peaks.empty:
        return df.nlargest(top_n, prob_col)

    sorted_df = df_peaks.sort_values(by=prob_col, ascending=False)
    selected_indices = []
    min_dist_frames = 4 * SAMPLING_RATE_HZ

    for idx, row in sorted_df.iterrows():
        f = row["Frame"]
        if all(abs(f - df.loc[s_idx, "Frame"]) > min_dist_frames for s_idx in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) >= top_n:
            break

    cols_to_keep = [c for c in [
        "Frame", "Time_s", "possession_team", prob_col, "feat_def_dist_1",
        "feat_closing_speed_max", "feat_pressure_index", "feat_def_hull_area", "Ball_x", "Ball_y", "tactical_outcome"
    ] if c in df.columns]

    return df.loc[selected_indices, cols_to_keep].reset_index(drop=True)
