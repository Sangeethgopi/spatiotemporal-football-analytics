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

def evaluate_events(y_true, y_prob, threshold, tolerance_frames=50, min_duration=0, merge_gap=1):
    y_pred = (y_prob >= threshold).astype(int)
    
    # Extract events with gap merging
    def get_events(arr):
        events = []
        in_event = False
        start = 0
        zero_gap = 0
        for i in range(len(arr)):
            if arr[i] == 1:
                if not in_event:
                    in_event = True
                    start = i
                zero_gap = 0
            elif arr[i] == 0 and in_event:
                zero_gap += 1
                if zero_gap > merge_gap:
                    in_event = False
                    if (i - zero_gap - start) >= min_duration:
                        events.append((start, i - zero_gap))
        if in_event:
            if (len(arr) - start) >= min_duration:
                events.append((start, len(arr)))
        return events
        
    def get_gt_events(arr):
        events = []
        in_event, start = False, 0
        for i in range(len(arr)):
            if arr[i] == 1 and not in_event:
                in_event, start = True, i
            elif arr[i] == 0 and in_event:
                in_event = False
                events.append((start, i))
        if in_event: events.append((start, len(arr)))
        return events
        
    pred_events = get_events(y_pred)
    true_events = get_gt_events(y_true)
    
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
    return {'f1': f1, 'prec': prec, 'rec': rec, 'tp': tp, 'fp': fp, 'fn': fn, 'n_pred': len(pred_events), 'n_gt': len(true_events)}

def evaluate_frame_metrics(y_true, y_prob):
    valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
    yt = y_true[valid].astype(int)
    yp = y_prob[valid]
    if len(np.unique(yt)) < 2: return {'roc': 0.5, 'pr': 0.0}
    roc = roc_auc_score(yt, yp)
    prec, rec, _ = precision_recall_curve(yt, yp)
    pr = auc(rec, prec)
    return {'roc': roc, 'pr': pr}

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
    
    # Add temporal features
    df_feat_temp = df_features.copy()
    w = 25
    df_feat_temp['feat_temp_accel'] = df_feat_temp['feat_closing_speed_max'].diff(w).fillna(0)
    df_feat_temp['feat_temp_hull_shrink'] = df_feat_temp['feat_def_hull_area'].diff(w).fillna(0)
    
    best_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']
    
    # Train test split
    df_train_full = df_feat_temp.iloc[:split_idx - 100].copy()
    df_test = df_feat_temp.iloc[split_idx:].copy()
    y_train = df_train_full['target_press_trigger'].values
    y_test = df_test['target_press_trigger'].values
    
    # 1. Baseline Test Evaluation
    X_tr_base = df_train_full[FEATURE_COLUMNS].fillna(0)
    X_te_base = df_test[FEATURE_COLUMNS].fillna(0)
    clf_base = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_base.fit(X_tr_base, y_train)
    p_test_base = clf_base.predict_proba(X_te_base)[:, 1]
    ev_base_test = evaluate_events(y_test, p_test_base, 0.50, merge_gap=1)
    fm_base_test = evaluate_frame_metrics(y_test, p_test_base)
    
    # 2. Chosen Model Test Evaluation
    X_tr_best = df_train_full[best_cols].fillna(0)
    X_te_best = df_test[best_cols].fillna(0)
    clf_best = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_best.fit(X_tr_best, y_train)
    p_test_best = clf_best.predict_proba(X_te_best)[:, 1]
    p_test_final = pd.Series(p_test_best).rolling(25, min_periods=1, center=False).mean().values
    ev_final_test = evaluate_events(y_test, p_test_final, 0.50, merge_gap=1)
    fm_final_test = evaluate_frame_metrics(y_test, p_test_final)
    
    # 3. Target Uncoupled
    turnovers_unc = derive_turnovers_uncoupled(df_poss)
    df_lab_unc = label_turnover_horizons(df_poss, turnovers_unc)
    y_test_unc = df_lab_unc.iloc[split_idx:]['target_press_trigger'].values
    ev_unc_test = evaluate_events(y_test_unc, p_test_final, 0.50, merge_gap=1)
    
    out_dir = "data/results/model_improvement"
    final_res = {
        'Baseline_Test': {**fm_base_test, **ev_base_test},
        'Candidate_Test': {**fm_final_test, **ev_final_test},
        'Uncoupled_Test': ev_unc_test,
    }
    with open(f"{out_dir}/MODEL_IMPROVEMENT_RESULTS.json", "w") as f:
        json.dump(final_res, f, indent=4)
        
    print("Test Ev F1 Base:", ev_base_test['f1'])
    print("Test Ev F1 Final:", ev_final_test['f1'])

if __name__ == '__main__':
    main()
