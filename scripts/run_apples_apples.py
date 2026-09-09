import os, json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS

def derive_turnovers_uncoupled(df_possession, min_hold_frames=75):
    teams = df_possession['possession_team'].values
    turnover_rows = []
    current_team = None
    team_hold_frames = 0
    for i in range(len(teams)):
        team = teams[i]
        if pd.isna(team):
            team_hold_frames = 0
            continue
        if team != current_team:
            if team_hold_frames >= min_hold_frames and current_team is not None:
                turnover_rows.append({'frame_idx': i, 'frame': i, 'winning_team': team})
            current_team = team
            team_hold_frames = 1
        else:
            team_hold_frames += 1
    return pd.DataFrame(turnover_rows)

def evaluate_events_strict(y_true, y_prob, threshold, tolerance_frames=50):
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
            if (p_start <= t_exp_end) and (p_end >= t_exp_start):
                tp_preds.add(p_idx)
                matched_true.add(t_idx)
                
    tp = len(matched_true) 
    fp = len(pred_events) - len(tp_preds)
    fn = len(true_events) - len(matched_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'f1': f1, 'prec': prec, 'rec': rec, 'tp': tp, 'fp': fp, 'fn': fn, 'n_pred': len(pred_events), 'n_gt': len(true_events), 'pred_events': pred_events, 'true_events': true_events}

def evaluate_frame_metrics(y_true, y_prob, threshold):
    valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
    yt = y_true[valid].astype(int)
    yp = y_prob[valid]
    y_pred = (yp >= threshold).astype(int)
    if len(np.unique(yt)) < 2: return {'roc': 0.5, 'pr': 0.0, 'f_prec': 0, 'f_rec': 0, 'f_f1': 0}
    roc = roc_auc_score(yt, yp)
    prec, rec, _ = precision_recall_curve(yt, yp)
    pr = auc(rec, prec)
    
    tp = np.sum((y_pred == 1) & (yt == 1))
    fp = np.sum((y_pred == 1) & (yt == 0))
    fn = np.sum((y_pred == 0) & (yt == 1))
    f_prec = tp / (tp + fp) if tp + fp > 0 else 0
    f_rec = tp / (tp + fn) if tp + fn > 0 else 0
    f_f1 = 2 * f_prec * f_rec / (f_prec + f_rec) if f_prec + f_rec > 0 else 0
    return {'roc': roc, 'pr': pr, 'f_prec': f_prec, 'f_rec': f_rec, 'f_f1': f_f1}

def get_target_horizon_metrics(pred_events, true_events):
    lead_times = []
    durations = []
    overlaps = []
    
    for p_start, p_end in pred_events:
        durations.append(p_end - p_start)
        
        # Match to GT to find overlap and lead time
        # GT is [t_start, t_end] where t_end is turnover
        # lead time = t_end - p_start
        matched_gt = None
        for t_start, t_end in true_events:
            t_exp_start = max(0, t_start - 50)
            t_exp_end = t_end + 50
            if (p_start <= t_exp_end) and (p_end >= t_exp_start):
                matched_gt = (t_start, t_end)
                break
        
        if matched_gt:
            t_start, t_end = matched_gt
            lead_times.append(t_end - p_start)
            overlap_start = max(p_start, t_start)
            overlap_end = min(p_end, t_end)
            if overlap_end > overlap_start:
                overlaps.append(overlap_end - overlap_start)
            else:
                overlaps.append(0)
                
    return {
        'median_lead_time': np.median(lead_times) if lead_times else 0,
        'median_duration': np.median(durations) if durations else 0,
        'median_overlap': np.median(overlaps) if overlaps else 0
    }

def main():
    print("Loading data...")
    paths = ensure_dataset_available('data', 'match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)
    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    df_features = extract_pressing_features(df_labeled)
    
    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    
    df_feat_temp = df_features.copy()
    w = 25
    df_feat_temp['feat_temp_accel'] = df_feat_temp['feat_closing_speed_max'].diff(w).fillna(0)
    df_feat_temp['feat_temp_hull_shrink'] = df_feat_temp['feat_def_hull_area'].diff(w).fillna(0)
    
    best_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']
    
    df_train_full = df_feat_temp.iloc[:split_idx - 100].copy()
    val_split_idx = int(len(df_train_full) * 0.8)
    df_train = df_train_full.iloc[:val_split_idx].copy()
    df_val = df_train_full.iloc[val_split_idx:].copy()
    df_test = df_feat_temp.iloc[split_idx:].copy()
    
    y_train = df_train['target_press_trigger'].values
    y_val = df_val['target_press_trigger'].values
    y_train_full = df_train_full['target_press_trigger'].values
    y_test = df_test['target_press_trigger'].values
    
    res = {}
    
    print("Evaluating Validation...")
    # Base 31
    clf_base = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_base.fit(df_train[FEATURE_COLUMNS].fillna(0), y_train)
    p_val_base = clf_base.predict_proba(df_val[FEATURE_COLUMNS].fillna(0))[:, 1]
    res['val_base31'] = evaluate_events_strict(y_val, p_val_base, 0.50)
    
    # 33 features
    clf_33 = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_33.fit(df_train[best_cols].fillna(0), y_train)
    p_val_33 = clf_33.predict_proba(df_val[best_cols].fillna(0))[:, 1]
    res['val_33'] = evaluate_events_strict(y_val, p_val_33, 0.50)
    
    # Smoothing sweeps
    smooth_windows = {'0s': 1, '0.5s': 12, '1.0s': 25, '1.5s': 37, '2.0s': 50, '3.0s': 75}
    res['val_smooth'] = {}
    for name, frms in smooth_windows.items():
        if frms == 1:
            p_sm = p_val_33
        else:
            p_sm = pd.Series(p_val_33).rolling(frms, min_periods=1, center=False).mean().values
        res['val_smooth'][name] = evaluate_events_strict(y_val, p_sm, 0.50)
        
    print("Evaluating Test...")
    # Baseline Test
    clf_base_test = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_base_test.fit(df_train_full[FEATURE_COLUMNS].fillna(0), y_train_full)
    p_test_base = clf_base_test.predict_proba(df_test[FEATURE_COLUMNS].fillna(0))[:, 1]
    ev_test_base = evaluate_events_strict(y_test, p_test_base, 0.50)
    fm_test_base = evaluate_frame_metrics(y_test, p_test_base, 0.50)
    
    # Winning Test
    clf_win = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_win.fit(df_train_full[best_cols].fillna(0), y_train_full)
    p_test_win_raw = clf_win.predict_proba(df_test[best_cols].fillna(0))[:, 1]
    p_test_win = pd.Series(p_test_win_raw).rolling(25, min_periods=1, center=False).mean().values
    ev_test_win = evaluate_events_strict(y_test, p_test_win, 0.50)
    fm_test_win = evaluate_frame_metrics(y_test, p_test_win, 0.50)
    
    res['test_baseline'] = {**fm_test_base, **ev_test_base}
    res['test_winning'] = {**fm_test_win, **ev_test_win}
    
    res['test_target_metrics'] = get_target_horizon_metrics(ev_test_win['pred_events'], ev_test_win['true_events'])
    
    # Uncoupled Target
    turnovers_unc = derive_turnovers_uncoupled(df_poss)
    df_lab_unc = label_turnover_horizons(df_poss, turnovers_unc)
    y_test_unc = df_lab_unc.iloc[split_idx:]['target_press_trigger'].values
    res['test_uncoupled'] = evaluate_events_strict(y_test_unc, p_test_win, 0.50)

    # Clean up non-serializable fields
    for d in [res['val_base31'], res['val_33']] + list(res['val_smooth'].values()) + [res['test_baseline'], res['test_winning'], res['test_uncoupled']]:
        d.pop('pred_events', None)
        d.pop('true_events', None)
        
    out_dir = "data/results/model_improvement"
    with open(f"{out_dir}/FINAL_APPLES_TO_APPLES_RESULTS.json", "w") as f:
        json.dump(res, f, indent=4)
        
    md = f"""# FINAL APPLES-TO-APPLES EVALUATION

## 1. TEST SET COMPARISON
| Metric | Baseline HistGB (31, Unsmoothed) | Winning Model (33, 1s Smoothed) |
| :--- | :--- | :--- |
| **ROC-AUC** | {res['test_baseline']['roc']:.3f} | {res['test_winning']['roc']:.3f} |
| **PR-AUC** | {res['test_baseline']['pr']:.3f} | {res['test_winning']['pr']:.3f} |
| **Frame Precision** | {res['test_baseline']['f_prec']:.3f} | {res['test_winning']['f_prec']:.3f} |
| **Frame Recall** | {res['test_baseline']['f_rec']:.3f} | {res['test_winning']['f_rec']:.3f} |
| **Frame F1** | {res['test_baseline']['f_f1']:.3f} | {res['test_winning']['f_f1']:.3f} |
| **Event Precision** | {res['test_baseline']['prec']:.3f} | {res['test_winning']['prec']:.3f} |
| **Event Recall** | {res['test_baseline']['rec']:.3f} | {res['test_winning']['rec']:.3f} |
| **Event F1** | {res['test_baseline']['f1']:.3f} | {res['test_winning']['f1']:.3f} |
| **TP** | {res['test_baseline']['tp']} | {res['test_winning']['tp']} |
| **FP** | {res['test_baseline']['fp']} | {res['test_winning']['fp']} |
| **FN** | {res['test_baseline']['fn']} | {res['test_winning']['fn']} |
| **Predicted Events** | {res['test_baseline']['n_pred']} | {res['test_winning']['n_pred']} |

## 2. VALIDATION (Strict merge_gap=0)
- **Base 31 Features:** Event F1 = {res['val_base31']['f1']:.3f}
- **Base 31 + Temp Feats:** Event F1 = {res['val_33']['f1']:.3f}
- **Base 31 + Temp + 1s Smooth:** Event F1 = {res['val_smooth']['1.0s']['f1']:.3f}

## 3. SMOOTHING SENSITIVITY (Validation)
| Smoothing Window | Precision | Recall | F1 | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0s | {res['val_smooth']['0s']['prec']:.3f} | {res['val_smooth']['0s']['rec']:.3f} | {res['val_smooth']['0s']['f1']:.3f} | {res['val_smooth']['0s']['tp']} | {res['val_smooth']['0s']['fp']} | {res['val_smooth']['0s']['fn']} |
| 0.5s | {res['val_smooth']['0.5s']['prec']:.3f} | {res['val_smooth']['0.5s']['rec']:.3f} | {res['val_smooth']['0.5s']['f1']:.3f} | {res['val_smooth']['0.5s']['tp']} | {res['val_smooth']['0.5s']['fp']} | {res['val_smooth']['0.5s']['fn']} |
| 1.0s | {res['val_smooth']['1.0s']['prec']:.3f} | {res['val_smooth']['1.0s']['rec']:.3f} | {res['val_smooth']['1.0s']['f1']:.3f} | {res['val_smooth']['1.0s']['tp']} | {res['val_smooth']['1.0s']['fp']} | {res['val_smooth']['1.0s']['fn']} |
| 1.5s | {res['val_smooth']['1.5s']['prec']:.3f} | {res['val_smooth']['1.5s']['rec']:.3f} | {res['val_smooth']['1.5s']['f1']:.3f} | {res['val_smooth']['1.5s']['tp']} | {res['val_smooth']['1.5s']['fp']} | {res['val_smooth']['1.5s']['fn']} |
| 2.0s | {res['val_smooth']['2.0s']['prec']:.3f} | {res['val_smooth']['2.0s']['rec']:.3f} | {res['val_smooth']['2.0s']['f1']:.3f} | {res['val_smooth']['2.0s']['tp']} | {res['val_smooth']['2.0s']['fp']} | {res['val_smooth']['2.0s']['fn']} |
| 3.0s | {res['val_smooth']['3.0s']['prec']:.3f} | {res['val_smooth']['3.0s']['rec']:.3f} | {res['val_smooth']['3.0s']['f1']:.3f} | {res['val_smooth']['3.0s']['tp']} | {res['val_smooth']['3.0s']['fp']} | {res['val_smooth']['3.0s']['fn']} |

## 4. TARGET-HORIZON ANALYSIS (Winning Model)
- **Median First-Crossing Lead Time:** {res['test_target_metrics']['median_lead_time']} frames ({(res['test_target_metrics']['median_lead_time']/25):.2f}s)
- **Median Predicted-Event Duration:** {res['test_target_metrics']['median_duration']} frames ({(res['test_target_metrics']['median_duration']/25):.2f}s)
- **Median Overlap with 100-frame GT Horizon:** {res['test_target_metrics']['median_overlap']} frames ({(res['test_target_metrics']['median_overlap']/25):.2f}s)

## 5. UNCOUPLED TARGET (Test)
- TP: {res['test_uncoupled']['tp']}
- FP: {res['test_uncoupled']['fp']}
- FN: {res['test_uncoupled']['fn']}
- Precision: {res['test_uncoupled']['prec']:.3f}
- Recall: {res['test_uncoupled']['rec']:.3f}
- F1: {res['test_uncoupled']['f1']:.3f}
"""
    with open(f"{out_dir}/FINAL_APPLES_TO_APPLES_RESULTS.md", "w") as f:
        f.write(md)

if __name__ == '__main__':
    main()
