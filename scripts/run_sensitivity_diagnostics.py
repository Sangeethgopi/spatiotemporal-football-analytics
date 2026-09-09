import os, json, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from scripts.run_apples_apples import evaluate_events_strict

def run_diagnostics():
    print("Loading data...")
    paths = ensure_dataset_available('data', 'match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)
    
    out_dir = "data/results/model_improvement"
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # PHASE 6: TARGET SENSITIVITY (2m, 3m, 4m, 5m)
    # ---------------------------------------------------------
    print("Running Target Sensitivity...")
    target_sens_res = {}
    
    def get_target(dist):
        turnovers = derive_turnover_events(df_poss, forced_turnover_threshold_m=dist)
        df_l = label_turnover_horizons(df_poss, turnovers, horizon_frames=100)
        return turnovers, df_l
    
    # We just need to know the prevalence and event count
    for d in [2.0, 3.0, 4.0, 5.0]:
        t_events, df_l = get_target(d)
        n_gt = len(t_events)
        pos_prev = df_l['target_press_trigger'].mean()
        target_sens_res[f"{int(d)}m"] = {"n_gt": n_gt, "prevalence": pos_prev}
        
    # Uncoupled
    turnovers_unc = derive_turnover_events(df_poss, forced_turnover_threshold_m=999.0)
    df_l_unc = label_turnover_horizons(df_poss, turnovers_unc, horizon_frames=100)
    target_sens_res["uncoupled"] = {"n_gt": len(turnovers_unc), "prevalence": df_l_unc['target_press_trigger'].mean()}

    with open(f"{out_dir}/TARGET_SENSITIVITY.json", "w") as f:
        json.dump(target_sens_res, f, indent=4)
        
    md_target = "# TARGET SENSITIVITY\n\n| Target | GT Events | Frame Prevalence |\n| --- | --- | --- |\n"
    for k, v in target_sens_res.items():
        md_target += f"| {k} | {v['n_gt']} | {v['prevalence']:.4f} |\n"
    with open(f"{out_dir}/TARGET_SENSITIVITY.md", "w") as f:
        f.write(md_target)

    # ---------------------------------------------------------
    # PHASE 7: TARGET-HORIZON SENSITIVITY
    # ---------------------------------------------------------
    print("Running Horizon Sensitivity...")
    horizon_res = {}
    turnovers_3m = derive_turnover_events(df_poss, forced_turnover_threshold_m=3.0)
    
    # Train the base model on 100-frames to get predictions, then evaluate against different GT horizons
    # Actually, the user says "evaluate 50, 75, 100, 125". We use the same model predictions against different GTs?
    # Or retrain? "Do NOT change the primary result. As a secondary sensitivity analysis, evaluate..."
    # We will just use the predictions of the winning model against different GTs to see how Event F1 changes.
    
    df_labeled = label_turnover_horizons(df_poss, turnovers_3m, horizon_frames=100)
    df_features = extract_pressing_features(df_labeled)
    
    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    df_feat_temp = df_features.copy()
    w = 25
    df_feat_temp['feat_temp_accel'] = df_feat_temp['feat_closing_speed_max'].diff(w).fillna(0)
    df_feat_temp['feat_temp_hull_shrink'] = df_feat_temp['feat_def_hull_area'].diff(w).fillna(0)
    best_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']
    
    df_train_full = df_feat_temp.iloc[:split_idx - 100].copy()
    df_test = df_feat_temp.iloc[split_idx:].copy()
    
    clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf.fit(df_train_full[best_cols].fillna(0), df_train_full['target_press_trigger'].values)
    p_test_raw = clf.predict_proba(df_test[best_cols].fillna(0))[:, 1]
    p_test_win = pd.Series(p_test_raw).rolling(25, min_periods=1, center=False).mean().values
    
    for hz in [50, 75, 100, 125]:
        df_l_hz = label_turnover_horizons(df_poss, turnovers_3m, horizon_frames=hz)
        y_test_hz = df_l_hz.iloc[split_idx:]['target_press_trigger'].values
        ev = evaluate_events_strict(y_test_hz, p_test_win, 0.50)
        ev.pop('pred_events', None)
        ev.pop('true_events', None)
        horizon_res[f"{hz} frames"] = ev
        
    with open(f"{out_dir}/TARGET_HORIZON_SENSITIVITY.json", "w") as f:
        json.dump(horizon_res, f, indent=4)
        
    md_hz = "# HORIZON SENSITIVITY\n\n| Horizon | GT Events | Event F1 | Precision | Recall | TP | FP | FN |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for k, v in horizon_res.items():
        md_hz += f"| {k} | {v['n_gt']} | {v['f1']:.3f} | {v['prec']:.3f} | {v['rec']:.3f} | {v['tp']} | {v['fp']} | {v['fn']} |\n"
    with open(f"{out_dir}/TARGET_HORIZON_SENSITIVITY.md", "w") as f:
        f.write(md_hz)

    # ---------------------------------------------------------
    # PHASE 9 & 10: BOOTSTRAP UNCERTAINTY & ERROR ANALYSIS
    # ---------------------------------------------------------
    print("Running Error Analysis & Bootstrap...")
    # Get exact events
    ev_full = evaluate_events_strict(df_test['target_press_trigger'].values, p_test_win, 0.50)
    pred_ev = ev_full['pred_events']
    true_ev = ev_full['true_events']
    
    # Map TPs
    t_matched = []
    p_matched = []
    for p_idx, (p_s, p_e) in enumerate(pred_ev):
        for t_idx, (t_s, t_e) in enumerate(true_ev):
            t_exp_s = max(0, t_s - 50)
            t_exp_e = t_e + 50
            if (p_s <= t_exp_e) and (p_e >= t_exp_s):
                t_matched.append(t_idx)
                p_matched.append(p_idx)
                
    error_list = []
    for i, (t_s, t_e) in enumerate(true_ev):
        status = "TP" if i in t_matched else "FN"
        lead_time = 0
        p_dur = 0
        if status == "TP":
            p_idx = p_matched[t_matched.index(i)]
            p_s, p_e = pred_ev[p_idx]
            lead_time = t_e - p_s
            p_dur = p_e - p_s
        error_list.append({
            "event_id": i,
            "status": status,
            "gt_turnover_frame": t_e,
            "gt_duration": t_e - t_s,
            "lead_time_frames": lead_time,
            "pred_duration": p_dur
        })
        
    for i, (p_s, p_e) in enumerate(pred_ev):
        if i not in p_matched:
            error_list.append({
                "event_id": f"FP_{i}",
                "status": "FP",
                "gt_turnover_frame": None,
                "gt_duration": None,
                "lead_time_frames": None,
                "pred_duration": p_e - p_s
            })
            
    with open(f"{out_dir}/EVENT_ERROR_ANALYSIS.json", "w") as f:
        json.dump(error_list, f, indent=4)
        
    md_err = "# EVENT ERROR ANALYSIS\n\n| ID | Status | GT Turnover | Lead Time (fr) | Pred Duration (fr) |\n| --- | --- | --- | --- | --- |\n"
    for e in error_list:
        md_err += f"| {e['event_id']} | {e['status']} | {e['gt_turnover_frame']} | {e['lead_time_frames']} | {e['pred_duration']} |\n"
    with open(f"{out_dir}/EVENT_ERROR_ANALYSIS.md", "w") as f:
        f.write(md_err)
        
    # Bootstrap
    n_gt = len(true_ev)
    tp_orig = len(set(t_matched))
    fn_orig = n_gt - tp_orig
    fp_orig = len(pred_ev) - len(set(p_matched))
    
    # We bootstrap the set of true events and predicted events?
    # Simple block bootstrap of the 19 events
    np.random.seed(42)
    f1s = []
    true_event_indices = np.arange(len(true_ev))
    for _ in range(1000):
        # Sample true events with replacement
        sample_t = np.random.choice(true_event_indices, size=len(true_ev), replace=True)
        # Calculate TP and FN based on the sample
        s_tp = sum(1 for idx in sample_t if idx in t_matched)
        s_fn = len(sample_t) - s_tp
        
        # Sample FP events? There are FP_orig FPs in the whole test set. 
        # A simple approximation for F1 uncertainty:
        s_fp = fp_orig # FPs are independent of GT events
        
        s_prec = s_tp / (s_tp + s_fp) if s_tp + s_fp > 0 else 0
        s_rec = s_tp / (s_tp + s_fn) if s_tp + s_fn > 0 else 0
        s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if s_prec + s_rec > 0 else 0
        f1s.append(s_f1)
        
    f1s = np.array(f1s)
    ci_lower, ci_upper = np.percentile(f1s, [2.5, 97.5])
    
    uncert = {
        "f1_mean": np.mean(f1s),
        "f1_median": np.median(f1s),
        "f1_ci_2.5": ci_lower,
        "f1_ci_97.5": ci_upper
    }
    with open(f"{out_dir}/UNCERTAINTY.json", "w") as f:
        json.dump(uncert, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    run_diagnostics()
