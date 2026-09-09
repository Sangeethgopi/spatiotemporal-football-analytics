import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc
from sklearn.calibration import calibration_curve

# Append project root to path
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from src.model import PressTriggerClassifier
from src.baseline_ppda import RollingPPDABaselineClassifier, DistanceBaselineClassifier

def evaluate_events_rigorous(y_true, y_prob, threshold, tolerance_frames=50):
    """
    Rigorous event evaluation logic that handles fragmentation correctly.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    def get_events(arr):
        events = []
        in_event = False
        start = 0
        for i in range(len(arr)):
            if arr[i] == 1 and not in_event:
                in_event = True
                start = i
            elif arr[i] == 0 and in_event:
                in_event = False
                events.append((start, i))
        if in_event:
            events.append((start, len(arr)))
        return events
        
    pred_events = get_events(y_pred)
    true_events = get_events(y_true)
    
    tp_preds = set()
    matched_true = set()
    
    for p_idx, (p_start, p_end) in enumerate(pred_events):
        for t_idx, (t_start, t_end) in enumerate(true_events):
            t_exp_start = max(0, t_start - tolerance_frames)
            t_exp_end = t_end + tolerance_frames
            
            # Overlap condition
            if (p_start <= t_exp_end) and (p_end >= t_exp_start):
                tp_preds.add(p_idx)
                matched_true.add(t_idx)
                
    tp_events_found = len(matched_true) # True events successfully detected
    fp_pred_events = len(pred_events) - len(tp_preds) # Predicted events that hit nothing
    fn_true_events = len(true_events) - len(matched_true) # True events missed
    
    evt_prec = tp_events_found / (tp_events_found + fp_pred_events) if (tp_events_found + fp_pred_events) > 0 else 0.0
    evt_rec = tp_events_found / (tp_events_found + fn_true_events) if (tp_events_found + fn_true_events) > 0 else 0.0
    evt_f1 = 2 * evt_prec * evt_rec / (evt_prec + evt_rec) if (evt_prec + evt_rec) > 0 else 0.0
    
    return {
        'threshold': threshold,
        'pred_frames': int(np.sum(y_pred)),
        'pred_events': len(pred_events),
        'true_events': len(true_events),
        'tp_events': tp_events_found,
        'fp_events': fp_pred_events,
        'fn_events': fn_true_events,
        'event_precision': evt_prec,
        'event_recall': evt_rec,
        'event_f1': evt_f1
    }

def run_rigorous_evaluation():
    out_dir = os.path.join('data', 'results', 'rigorous_eval')
    os.makedirs(out_dir, exist_ok=True)
    sweep_dir = os.path.join(out_dir, 'threshold_sweeps')
    os.makedirs(sweep_dir, exist_ok=True)
    
    print("Loading Dataset...")
    paths = ensure_dataset_available(data_dir='data', match_id='match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)
    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    df_features = extract_pressing_features(df_labeled)
    
    # 6. Audit Event Definition
    print("Auditing Event Definitions...")
    y_target = df_features['target_press_trigger'].values
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(y_target):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append(i - start)
    if in_event:
        events.append(len(y_target) - start)
        
    audit_dict = {
        'underlying_turnover_events': len(turnovers_df),
        'total_positive_frames': int(np.sum(y_target)),
        'positive_windows_count': len(events),
        'min_event_duration_frames': int(np.min(events)) if events else 0,
        'median_event_duration_frames': float(np.median(events)) if events else 0,
        'max_event_duration_frames': int(np.max(events)) if events else 0
    }
    with open(os.path.join(out_dir, 'event_audit.json'), 'w') as f:
        json.dump(audit_dict, f, indent=4)
        
    # Purged Split
    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    df_train = df_features.iloc[:split_idx - 100].copy()
    df_val = df_features.iloc[split_idx:].copy()
    y_train = df_train['target_press_trigger'].values
    y_val = df_val['target_press_trigger'].values
    
    # Define Feature Sets
    spatial_cols = ['feat_def_dist_1', 'feat_def_dist_2', 'feat_def_dist_3', 'feat_def_dist_mean3', 'feat_def_density_5m', 'feat_def_density_10m', 'feat_att_density_10m', 'feat_numerical_advantage_10m', 'feat_passing_lane_occlusion', 'feat_viable_passing_options', 'feat_dist_to_goal', 'feat_dist_to_sideline', 'feat_ball_x', 'feat_ball_y']
    kinematic_cols = ['feat_closing_speed_max', 'feat_closing_speed_mean3', 'feat_ball_speed', 'feat_ball_progression_x', 'feat_roll_closing_speed_mean']
    shape_cols = ['feat_def_hull_area', 'feat_def_block_width', 'feat_def_block_depth', 'feat_def_dispersion', 'feat_def_centroid_dist_to_ball', 'feat_def_line_height', 'feat_roll_hull_area_change']
    
    feature_groups = {
        'Spatial': spatial_cols,
        'Shape': shape_cols,
        'Kinematic': kinematic_cols,
        'Spatial_Kinematic': spatial_cols + kinematic_cols,
        'Spatial_Shape': spatial_cols + shape_cols,
        'Kinematic_Shape': kinematic_cols + shape_cols,
        'Combined': spatial_cols + kinematic_cols + shape_cols
    }
    
    thresholds = np.linspace(0.05, 0.95, 19)
    pr_roc_data = []
    
    # ML Models
    for name, cols in feature_groups.items():
        print(f"Training {name}...")
        clf = PressTriggerClassifier(model_type='hist_gb')
        clf.feature_names = cols
        clf.fit(df_train, y_train)
        y_prob = clf.predict_proba(df_val)[:, 1]
        
        # Save PR / ROC / Calibration
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
        
        pr_roc_data.append({
            'model': name,
            'roc_auc': roc_auc_score(y_val, y_prob),
            'pr_auc': auc(rec, prec),
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'precision': prec.tolist(), 'recall': rec.tolist(),
            'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()
        })
        
        # Threshold Sweep
        sweep_results = []
        for thresh in thresholds:
            # Frame metrics
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))
            frm_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            frm_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            frm_f1 = 2 * frm_prec * frm_rec / (frm_prec + frm_rec) if (frm_prec + frm_rec) > 0 else 0
            
            evt_res = evaluate_events_rigorous(y_val, y_prob, thresh)
            evt_res.update({
                'frame_precision': frm_prec,
                'frame_recall': frm_rec,
                'frame_f1': frm_f1
            })
            sweep_results.append(evt_res)
            
        pd.DataFrame(sweep_results).to_csv(os.path.join(sweep_dir, f"{name}_sweep.csv"), index=False)
        
    # Logistic Regression Baseline
    print("Training Logistic Baseline...")
    clf_log = PressTriggerClassifier(model_type='logistic')
    clf_log.feature_names = feature_groups['Combined']
    clf_log.fit(df_train, y_train)
    y_prob_log = clf_log.predict_proba(df_val)[:, 1]
    
    sweep_log = []
    for thresh in thresholds:
        y_pred = (y_prob_log >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))
        frm_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        frm_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        frm_f1 = 2 * frm_prec * frm_rec / (frm_prec + frm_rec) if (frm_prec + frm_rec) > 0 else 0
        evt_res = evaluate_events_rigorous(y_val, y_prob_log, thresh)
        evt_res.update({'frame_precision': frm_prec, 'frame_recall': frm_rec, 'frame_f1': frm_f1})
        sweep_log.append(evt_res)
    pd.DataFrame(sweep_log).to_csv(os.path.join(sweep_dir, "Logistic_Combined_sweep.csv"), index=False)

    # Distance Sweep (1m, 2m, 3m, 4m, 5m, 10m)
    print("Running Distance Sweeps...")
    dist_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    dist_results = []
    for d in dist_thresholds:
        b_dist = DistanceBaselineClassifier(dist_threshold=d)
        y_prob_dist = b_dist.predict_proba(df_val)[:, 1]
        
        y_pred = (y_prob_dist >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y_val == 1))
        fp = np.sum((y_pred == 1) & (y_val == 0))
        fn = np.sum((y_pred == 0) & (y_val == 1))
        frm_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        frm_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        frm_f1 = 2 * frm_prec * frm_rec / (frm_prec + frm_rec) if (frm_prec + frm_rec) > 0 else 0
        
        evt_res = evaluate_events_rigorous(y_val, y_prob_dist, threshold=0.5)
        evt_res.update({
            'distance_threshold_m': d,
            'frame_precision': frm_prec,
            'frame_recall': frm_rec,
            'frame_f1': frm_f1
        })
        dist_results.append(evt_res)
    pd.DataFrame(dist_results).to_csv(os.path.join(out_dir, "distance_baseline_sweep.csv"), index=False)

    # Save PR / ROC data
    with open(os.path.join(out_dir, 'pr_roc_calibration.json'), 'w') as f:
        json.dump(pr_roc_data, f)
        
    print(f"Rigorous evaluation complete. Output saved to {out_dir}")

if __name__ == '__main__':
    run_rigorous_evaluation()
