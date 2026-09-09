import os, sys, json
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from src.model import PressTriggerClassifier
from src.baseline_ppda import RollingPPDABaselineClassifier, DistanceBaselineClassifier

def evaluate_events_detection(y_true, y_prob, threshold, tolerance_frames=50):
    """
    Event detection metric (NOT segmentation).
    Multiple fragmented predictions hitting the same true event count as one detection.
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
        if in_event: events.append((start, len(arr)))
        return events
        
    pred_events = get_events(y_pred)
    true_events = get_events(y_true)
    
    tp_preds = set()
    matched_true = set()
    
    for p_idx, (p_start, p_end) in enumerate(pred_events):
        for t_idx, (t_start, t_end) in enumerate(true_events):
            t_exp_start = max(0, t_start - tolerance_frames)
            t_exp_end = t_end + tolerance_frames
            if (p_start <= t_exp_end) and (p_end >= t_exp_start):
                tp_preds.add(p_idx)
                matched_true.add(t_idx)
                
    tp_events_found = len(matched_true) 
    fp_pred_events = len(pred_events) - len(tp_preds)
    fn_true_events = len(true_events) - len(matched_true)
    
    evt_prec = tp_events_found / (tp_events_found + fp_pred_events) if (tp_events_found + fp_pred_events) > 0 else 0.0
    evt_rec = tp_events_found / (tp_events_found + fn_true_events) if (tp_events_found + fn_true_events) > 0 else 0.0
    evt_f1 = 2 * evt_prec * evt_rec / (evt_prec + evt_rec) if (evt_prec + evt_rec) > 0 else 0.0
    
    # Frame metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    f_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec) if (f_prec + f_rec) > 0 else 0.0
    
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
        'event_f1': evt_f1,
        'frame_precision': f_prec,
        'frame_recall': f_rec,
        'frame_f1': f_f1
    }, pred_events, true_events, matched_true, tp_preds

def derive_turnovers_uncoupled(df_possession, min_hold_frames=75):
    teams = df_possession['possession_team'].values
    turnover_rows = []
    current_team = None
    current_start_idx = 0
    for i in range(len(teams)):
        t = teams[i]
        if current_team is None:
            current_team = t
            current_start_idx = i
        elif t != current_team:
            prev_duration_frames = i - current_start_idx
            if prev_duration_frames >= min_hold_frames:
                loss_idx = max(0, i - 50)
                turnover_rows.append({
                    'frame_idx': loss_idx,
                    'winning_team': t,
                    'prior_possession_duration_s': prev_duration_frames / 25.0
                })
            current_team = t
            current_start_idx = i
    return pd.DataFrame(turnover_rows)

def run_authoritative():
    print('Loading data...')
    paths = ensure_dataset_available('data', 'match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)

    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    df_features = extract_pressing_features(df_labeled)

    turnovers_uncoupled = derive_turnovers_uncoupled(df_poss)
    df_labeled_unc = label_turnover_horizons(df_poss, turnovers_uncoupled)
    df_features['target_uncoupled'] = df_labeled_unc['target_press_trigger']

    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    df_train = df_features.iloc[:split_idx - 100].copy()
    df_val = df_features.iloc[split_idx:].copy()

    y_train = df_train['target_press_trigger'].values
    y_val = df_val['target_press_trigger'].values
    y_val_uncoupled = df_val['target_uncoupled'].values
    
    counts = {
        'total_frames': n_frames,
        'train_frames': len(df_train),
        'test_frames': len(df_val),
        'purge_frames': 100
    }

    # Primary Model
    print('Training Combined Model...')
    clf_combined = PressTriggerClassifier(model_type='hist_gb')
    clf_combined.feature_names = FEATURE_COLUMNS
    clf_combined.fit(df_train, y_train)
    prob_combined = clf_combined.predict_proba(df_val)[:, 1]

    # Primary metrics
    res_primary, pred_ev, true_ev, mat_t, tp_p = evaluate_events_detection(y_val, prob_combined, 0.50, 50)
    fpr, tpr, _ = roc_curve(y_val, prob_combined)
    prec, rec, _ = precision_recall_curve(y_val, prob_combined)
    res_primary['roc_auc'] = roc_auc_score(y_val, prob_combined)
    res_primary['pr_auc'] = auc(rec, prec)
    res_primary['brier'] = brier_score_loss(y_val, prob_combined)

    # Baselines
    print('Training Baselines...')
    clf_ppda = RollingPPDABaselineClassifier(window_sec=15.0)
    clf_ppda.fit(df_train, y_train)
    prob_ppda = clf_ppda.predict_proba(df_val)[:, 1]
    res_ppda, _, _, _, _ = evaluate_events_detection(y_val, prob_ppda, 0.50, 50)
    
    clf_dist = DistanceBaselineClassifier(dist_threshold=3.0)
    prob_dist = clf_dist.predict_proba(df_val)[:, 1]
    res_dist, _, _, _, _ = evaluate_events_detection(y_val, prob_dist, 0.50, 50)

    # Ablations
    print('Running Ablations...')
    spatial_cols = ['feat_def_dist_1', 'feat_def_dist_2', 'feat_def_dist_3', 'feat_def_dist_mean3', 'feat_def_density_5m', 'feat_def_density_10m', 'feat_att_density_10m', 'feat_numerical_advantage_10m', 'feat_passing_lane_occlusion', 'feat_viable_passing_options', 'feat_dist_to_goal', 'feat_dist_to_sideline', 'feat_ball_x', 'feat_ball_y']
    shape_cols = ['feat_def_hull_area', 'feat_def_block_width', 'feat_def_block_depth', 'feat_def_dispersion', 'feat_def_centroid_dist_to_ball', 'feat_def_line_height', 'feat_roll_hull_area_change']
    kinematic_cols = ['feat_closing_speed_max', 'feat_closing_speed_mean3', 'feat_ball_speed', 'feat_ball_progression_x', 'feat_roll_closing_speed_mean']
    
    ablations = {}
    for name, cols in [('Spatial', spatial_cols), ('Shape', shape_cols), ('Kinematic', kinematic_cols)]:
        clf = PressTriggerClassifier(model_type='hist_gb')
        clf.feature_names = cols
        clf.fit(df_train, y_train)
        p_abl = clf.predict_proba(df_val)[:, 1]
        r, _, _, _, _ = evaluate_events_detection(y_val, p_abl, 0.50, 50)
        ablations[name] = r

    # Threshold Sensitivity (Descriptive)
    print('Running Sensitivity Analysis...')
    thresh_sens = []
    for th in np.linspace(0.05, 0.95, 19):
        r, _, _, _, _ = evaluate_events_detection(y_val, prob_combined, th, 50)
        thresh_sens.append(r)

    # Smoothing Sensitivity
    smooth_sens = {}
    for sec in [0.5, 1.0, 1.5, 2.0, 3.0]:
        frames = int(sec * 25)
        p_smooth = pd.Series(prob_combined).rolling(frames, min_periods=1).mean().values
        r, _, _, _, _ = evaluate_events_detection(y_val, p_smooth, 0.50, 50)
        smooth_sens[f'{sec}s'] = r

    # Uncoupled Target Sensitivity
    r_comb_unc, _, _, _, _ = evaluate_events_detection(y_val_uncoupled, prob_combined, 0.50, 50)
    r_ppda_unc, _, _, _, _ = evaluate_events_detection(y_val_uncoupled, prob_ppda, 0.50, 50)
    r_dist_unc, _, _, _, _ = evaluate_events_detection(y_val_uncoupled, prob_dist, 0.50, 50)
    
    clf_sp_only = PressTriggerClassifier(model_type='hist_gb')
    clf_sp_only.feature_names = spatial_cols
    clf_sp_only.fit(df_train, y_train)
    p_sp = clf_sp_only.predict_proba(df_val)[:, 1]
    r_sp_unc, _, _, _, _ = evaluate_events_detection(y_val_uncoupled, p_sp, 0.50, 50)
    
    uncoupled_sens = {
        'Combined': r_comb_unc,
        'PPDA': r_ppda_unc,
        'Distance3m': r_dist_unc,
        'Spatial': r_sp_unc
    }

    # Block Variability
    blocks = []
    n_blocks = 10
    block_sz = len(y_val) // n_blocks
    for i in range(n_blocks):
        s = i * block_sz
        e = (i+1)*block_sz if i < n_blocks - 1 else len(y_val)
        r, _, _, _, _ = evaluate_events_detection(y_val[s:e], prob_combined[s:e], 0.50, 50)
        blocks.append(r)

    # Error Topologies
    errors = {'TP': [], 'FP': [], 'FN': []}
    for p_idx in list(tp_p)[:10]:
        start, end = pred_ev[p_idx]
        errors['TP'].append({'frames': [start, end], 'hull': df_val['feat_def_hull_area'].iloc[start:end].mean(), 'dist': df_val['feat_def_dist_1'].iloc[start:end].mean()})
    fp_idx = [i for i in range(len(pred_ev)) if i not in tp_p]
    for p_idx in fp_idx[:10]:
        start, end = pred_ev[p_idx]
        errors['FP'].append({'frames': [start, end], 'hull': df_val['feat_def_hull_area'].iloc[start:end].mean(), 'dist': df_val['feat_def_dist_1'].iloc[start:end].mean()})
    fn_idx = [i for i in range(len(true_ev)) if i not in mat_t]
    for t_idx in fn_idx[:10]:
        start, end = true_ev[t_idx]
        errors['FN'].append({'frames': [start, end], 'hull': df_val['feat_def_hull_area'].iloc[start:end].mean(), 'dist': df_val['feat_def_dist_1'].iloc[start:end].mean()})

    out = {
        'counts': counts,
        'primary': res_primary,
        'ppda': res_ppda,
        'distance': res_dist,
        'ablations': ablations,
        'threshold_sensitivity': thresh_sens,
        'smoothing_sensitivity': smooth_sens,
        'uncoupled': uncoupled_sens,
        'block_variability': blocks,
        'errors': errors,
        'cross_match': 'untested'
    }

    with open('data/results/rigorous_eval/AUTHORITATIVE_RESULTS.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Authoritative Evaluation Complete.')

if __name__ == '__main__':
    run_authoritative()
