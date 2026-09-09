import os, json, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
                turnover_rows.append({'frame_idx': i, 'frame': i, 'turnover_team': team})
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
        
    pred_events = get_events(y_pred)
    # Ground truth events are always perfectly contiguous blocks
    
    # Special: For GT we don't merge or filter duration, just extract
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
    
    # Chronological Train/Val/Test
    df_train_full = df_features.iloc[:split_idx - 100].copy()
    val_split_idx = int(len(df_train_full) * 0.8)
    
    df_train = df_train_full.iloc[:val_split_idx].copy()
    df_val = df_train_full.iloc[val_split_idx:].copy()
    df_test = df_features.iloc[split_idx:].copy()
    
    y_train = df_train['target_press_trigger'].values
    y_val = df_val['target_press_trigger'].values
    y_test = df_test['target_press_trigger'].values
    
    out_dir = "data/results/model_improvement"
    os.makedirs(out_dir, exist_ok=True)
    
    # ============================================================
    # PHASE 1: FEATURE ANALYSIS
    # ============================================================
    print("Phase 1: Feature Analysis")
    feat_stats = []
    for col in FEATURE_COLUMNS:
        series = df_train_full[col]
        valid = series.dropna()
        miss = series.isna().mean()
        corr = df_train_full[['target_press_trigger', col]].corr().iloc[0, 1] if valid.nunique() > 1 else 0
        feat_stats.append({
            'Feature': col,
            'Missing_Pct': miss,
            'Mean': valid.mean(),
            'Std': valid.std(),
            'Min': valid.min(),
            'Max': valid.max(),
            'Corr_with_Target': corr
        })
    pd.DataFrame(feat_stats).to_csv(f"{out_dir}/FEATURE_QUALITY.csv", index=False)
    
    # ============================================================
    # PHASE 2: FEATURE ABLATION
    # ============================================================
    print("Phase 2: Feature Ablation")
    spatial = ['feat_def_dist_1', 'feat_def_dist_2', 'feat_def_dist_3', 'feat_def_dist_mean3', 'feat_def_density_5m', 'feat_def_density_10m', 'feat_att_density_10m', 'feat_numerical_advantage_10m', 'feat_passing_lane_occlusion', 'feat_viable_passing_options', 'feat_dist_to_goal', 'feat_dist_to_sideline', 'feat_ball_x', 'feat_ball_y']
    kinematic = ['feat_closing_speed_max', 'feat_closing_speed_mean3', 'feat_ball_speed', 'feat_ball_progression_x', 'feat_roll_closing_speed_mean']
    shape = ['feat_def_hull_area', 'feat_def_block_width', 'feat_def_block_depth', 'feat_def_dispersion', 'feat_def_centroid_dist_to_ball', 'feat_def_line_height', 'feat_roll_hull_area_change']
    temporal = ['feat_pressure_index', 'feat_roll_pressure_index_mean', 'feat_roll_pressure_index_max', 'feat_roll_dist_mean3_min', 'feat_possession_duration_s']
    
    groups = {
        'Spatial': spatial,
        'Kinematic': kinematic,
        'Shape': shape,
        'Spatial+Shape': spatial + shape,
        'Spatial+Kinematic': spatial + kinematic,
        'Shape+Kinematic': shape + kinematic,
        'All_31': FEATURE_COLUMNS
    }
    
    ablation_res = []
    for gname, cols in groups.items():
        clf = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
        X_tr = df_train[cols].fillna(0.0)
        X_va = df_val[cols].fillna(0.0)
        clf.fit(X_tr, y_train)
        p_val = clf.predict_proba(X_va)[:, 1]
        ev = evaluate_events(y_val, p_val, 0.50, merge_gap=1)
        fm = evaluate_frame_metrics(y_val, p_val)
        ablation_res.append({'Group': gname, 'ROC': fm['roc'], 'PR': fm['pr'], 'Ev_Prec': ev['prec'], 'Ev_Rec': ev['rec'], 'Ev_F1': ev['f1']})
    pd.DataFrame(ablation_res).to_csv(f"{out_dir}/FEATURE_ABLATION_RESULTS.csv", index=False)
    
    # ============================================================
    # PHASE 3: HISTGB OPTIMISATION
    # ============================================================
    print("Phase 3: HistGB Tuning")
    configs = [
        {'lr': 0.04, 'mln': 15, 'l2': 1.5},
        {'lr': 0.02, 'mln': 15, 'l2': 1.5},
        {'lr': 0.10, 'mln': 31, 'l2': 0.5},
        {'lr': 0.04, 'mln': 7,  'l2': 3.0}
    ]
    tune_res = []
    best_hgb_f1 = -1
    best_hgb_prob = None
    best_hgb_cfg = None
    X_tr_all = df_train[FEATURE_COLUMNS].fillna(0.0)
    X_va_all = df_val[FEATURE_COLUMNS].fillna(0.0)
    
    for cfg in configs:
        clf = HistGradientBoostingClassifier(learning_rate=cfg['lr'], max_leaf_nodes=cfg['mln'], l2_regularization=cfg['l2'], random_state=42)
        clf.fit(X_tr_all, y_train)
        p_val = clf.predict_proba(X_va_all)[:, 1]
        ev = evaluate_events(y_val, p_val, 0.50, merge_gap=1)
        tune_res.append({'LR': cfg['lr'], 'MaxLeaves': cfg['mln'], 'L2': cfg['l2'], 'Ev_F1': ev['f1']})
        if ev['f1'] > best_hgb_f1:
            best_hgb_f1 = ev['f1']
            best_hgb_prob = p_val
            best_hgb_cfg = cfg
            best_hgb_model = clf
    pd.DataFrame(tune_res).to_csv(f"{out_dir}/MODEL_SEARCH_RESULTS.csv", index=False)
    
    # ============================================================
    # PHASE 4: CLASSICAL MODELS
    # ============================================================
    print("Phase 4: Classical Models")
    pipe_lr = Pipeline([('imp', SimpleImputer()), ('scl', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])
    pipe_rf = Pipeline([('imp', SimpleImputer()), ('clf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))])
    
    class_res = []
    for mname, model in [('HistGB(Best)', best_hgb_model), ('LogReg', pipe_lr), ('RandomForest', pipe_rf)]:
        model.fit(X_tr_all, y_train)
        p_val = model.predict_proba(X_va_all)[:, 1]
        ev = evaluate_events(y_val, p_val, 0.50, merge_gap=1)
        fm = evaluate_frame_metrics(y_val, p_val)
        class_res.append({'Model': mname, 'ROC': fm['roc'], 'PR': fm['pr'], 'Ev_F1': ev['f1']})
    pd.DataFrame(class_res).to_csv(f"{out_dir}/CLASSICAL_MODEL_COMPARISON.csv", index=False)
    
    # ============================================================
    # PHASE 6: TEMPORAL FEATURE ENGINEERING
    # ============================================================
    print("Phase 6: Temporal Features")
    # Add backward derivatives
    df_feat_temp = df_features.copy()
    w = 25 # 1 second backward
    df_feat_temp['feat_temp_accel'] = df_feat_temp['feat_closing_speed_max'].diff(w).fillna(0)
    df_feat_temp['feat_temp_hull_shrink'] = df_feat_temp['feat_def_hull_area'].diff(w).fillna(0)
    
    df_tr_t = df_feat_temp.iloc[:split_idx - 100].iloc[:val_split_idx].copy()
    df_va_t = df_feat_temp.iloc[:split_idx - 100].iloc[val_split_idx:].copy()
    new_cols = FEATURE_COLUMNS + ['feat_temp_accel', 'feat_temp_hull_shrink']
    
    clf_t = HistGradientBoostingClassifier(learning_rate=best_hgb_cfg['lr'], max_leaf_nodes=best_hgb_cfg['mln'], l2_regularization=best_hgb_cfg['l2'], random_state=42)
    clf_t.fit(df_tr_t[new_cols].fillna(0), y_train)
    p_val_t = clf_t.predict_proba(df_va_t[new_cols].fillna(0))[:, 1]
    ev_t = evaluate_events(y_val, p_val_t, 0.50, merge_gap=1)
    pd.DataFrame([{'Features': 'Base_31', 'Ev_F1': best_hgb_f1}, {'Features': 'Base+Derivatives', 'Ev_F1': ev_t['f1']}]).to_csv(f"{out_dir}/TEMPORAL_FEATURE_RESULTS.csv", index=False)
    
    best_val_prob = p_val_t if ev_t['f1'] > best_hgb_f1 else best_hgb_prob
    best_cols = new_cols if ev_t['f1'] > best_hgb_f1 else FEATURE_COLUMNS
    best_model = clf_t if ev_t['f1'] > best_hgb_f1 else best_hgb_model
    
    # ============================================================
    # PHASE 7 & 8: THRESHOLD & POST-PROCESSING
    # ============================================================
    print("Phase 7 & 8: Post-processing & Thresholds")
    pp_res = []
    # Test thresholds on unsmoothed
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ev = evaluate_events(y_val, best_val_prob, th, merge_gap=1)
        pp_res.append({'Smoothing': 'None', 'Thresh': th, 'Ev_F1': ev['f1'], 'Ev_Prec': ev['prec'], 'Ev_Rec': ev['rec']})
        
    # Test rolling smooth (1.0s = 25 frames)
    smooth_prob = pd.Series(best_val_prob).rolling(25, min_periods=1, center=False).mean().values
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ev = evaluate_events(y_val, smooth_prob, th, merge_gap=1) # Note gap logic still applies
        pp_res.append({'Smoothing': '1.0s', 'Thresh': th, 'Ev_F1': ev['f1'], 'Ev_Prec': ev['prec'], 'Ev_Rec': ev['rec']})
        
    pd.DataFrame(pp_res).to_csv(f"{out_dir}/POSTPROCESSING_RESULTS.csv", index=False)
    
    # Select best combination purely from Val
    best_pp = max(pp_res, key=lambda x: x['Ev_F1'])
    print(f"Best Val config: Smooth={best_pp['Smoothing']}, Thresh={best_pp['Thresh']}, F1={best_pp['Ev_F1']}")
    
    # ============================================================
    # PHASE 5: TEMPORAL GRU (Simple) & PHASE 9 ENSEMBLE
    # ============================================================
    print("Phase 5: GRU")
    class SimpleGRU(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.gru = nn.GRU(input_dim, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])
            
    seq_len = 50
    scaler = StandardScaler()
    s_tr = scaler.fit_transform(df_train[FEATURE_COLUMNS].fillna(0))
    s_va = scaler.transform(df_val[FEATURE_COLUMNS].fillna(0))
    
    def create_seq(X, y, seq):
        Xs, ys = [], []
        for i in range(seq-1, len(X)):
            Xs.append(X[i-seq+1:i+1])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
        
    X_seq_tr, y_seq_tr = create_seq(s_tr, y_train, seq_len)
    X_seq_va, y_seq_va = create_seq(s_va, y_val, seq_len)
    
    dataset = TensorDataset(torch.tensor(X_seq_tr, dtype=torch.float32), torch.tensor(y_seq_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    gru = SimpleGRU(len(FEATURE_COLUMNS))
    opt = torch.optim.Adam(gru.parameters(), lr=0.002)
    pos_w = torch.tensor([sum(y_seq_tr==0)/sum(y_seq_tr==1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    gru.train()
    for ep in range(3):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(gru(bx), by)
            loss.backward()
            opt.step()
            
    gru.eval()
    with torch.no_grad():
        logits_va = gru(torch.tensor(X_seq_va, dtype=torch.float32))
        p_gru_va = torch.sigmoid(logits_va).numpy().flatten()
        
    # Pad GRU predictions with 0s for the first seq_len-1 frames
    p_gru_va_full = np.concatenate([np.zeros(seq_len-1), p_gru_va])
    ev_gru = evaluate_events(y_val, p_gru_va_full, 0.50)
    
    pd.DataFrame([{'Model': 'GRU_50', 'Ev_F1': ev_gru['f1']}]).to_csv(f"{out_dir}/TEMPORAL_MODEL_COMPARISON.csv", index=False)
    
    # Ensemble Phase 9
    ens_res = []
    for w_hgb in [0.25, 0.50, 0.75]:
        p_ens = w_hgb * best_val_prob + (1 - w_hgb) * p_gru_va_full
        ev_e = evaluate_events(y_val, p_ens, 0.50)
        ens_res.append({'W_HistGB': w_hgb, 'Ev_F1': ev_e['f1']})
    pd.DataFrame(ens_res).to_csv(f"{out_dir}/ENSEMBLE_RESULTS.csv", index=False)
    
    best_ens = max(ens_res, key=lambda x: x['Ev_F1'])
    
    # Decide Final Selection
    # If Ensemble is better than best_pp Ev_F1, pick Ensemble, else pick best_pp HistGB
    use_ensemble = False
    if best_ens['Ev_F1'] > best_pp['Ev_F1']:
        use_ensemble = True
        print("Ensemble won on Val!")
    else:
        print("HistGB won on Val!")
        
    # ============================================================
    # PHASE 10 & 11: FINAL TEST EVALUATION
    # ============================================================
    print("Phase 10: Final Test Evaluation")
    # Retrain final baseline and chosen model on FULL TRAIN (train+val) before test? 
    # NO! Keep strictly to the trained instances to avoid bleeding or changing distributions.
    
    # 1. Baseline Test Evaluation
    X_te_all = df_test[FEATURE_COLUMNS].fillna(0)
    clf_base = HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.5, random_state=42)
    clf_base.fit(df_train_full[FEATURE_COLUMNS].fillna(0), df_train_full['target_press_trigger'].values)
    p_test_base = clf_base.predict_proba(X_te_all)[:, 1]
    ev_base_test = evaluate_events(y_test, p_test_base, 0.50, merge_gap=1)
    fm_base_test = evaluate_frame_metrics(y_test, p_test_base)
    
    # 2. Chosen Model Test Evaluation
    X_tr_full_best = df_feat_temp.iloc[:split_idx - 100][best_cols].fillna(0)
    X_te_best = df_feat_temp.iloc[split_idx:][best_cols].fillna(0)
    best_model.fit(X_tr_full_best, df_train_full['target_press_trigger'].values)
    p_test_best = best_model.predict_proba(X_te_best)[:, 1]
    
    if best_pp['Smoothing'] == '1.0s':
        p_test_best = pd.Series(p_test_best).rolling(25, min_periods=1, center=False).mean().values
        
    if use_ensemble:
        # Train GRU on full train
        s_tr_full = scaler.fit_transform(X_tr_full_best[FEATURE_COLUMNS])
        s_te = scaler.transform(X_te_best[FEATURE_COLUMNS])
        X_seq_tr_full, y_seq_tr_full = create_seq(s_tr_full, df_train_full['target_press_trigger'].values, seq_len)
        X_seq_te, _ = create_seq(s_te, y_test, seq_len)
        
        dset_full = TensorDataset(torch.tensor(X_seq_tr_full, dtype=torch.float32), torch.tensor(y_seq_tr_full, dtype=torch.float32).unsqueeze(1))
        loader_full = DataLoader(dset_full, batch_size=256, shuffle=True)
        gru_full = SimpleGRU(len(FEATURE_COLUMNS))
        opt_f = torch.optim.Adam(gru_full.parameters(), lr=0.002)
        pos_w_f = torch.tensor([sum(y_seq_tr_full==0)/sum(y_seq_tr_full==1)])
        crit_f = nn.BCEWithLogitsLoss(pos_weight=pos_w_f)
        
        gru_full.train()
        for ep in range(3):
            for bx, by in loader_full:
                opt_f.zero_grad()
                loss = crit_f(gru_full(bx), by)
                loss.backward()
                opt_f.step()
                
        gru_full.eval()
        with torch.no_grad():
            p_gru_te = torch.sigmoid(gru_full(torch.tensor(X_seq_te, dtype=torch.float32))).numpy().flatten()
        p_gru_te_full = np.concatenate([np.zeros(seq_len-1), p_gru_te])
        
        w_hgb = best_ens['W_HistGB']
        p_test_final = w_hgb * p_test_best + (1 - w_hgb) * p_gru_te_full
        final_th = 0.50
    else:
        p_test_final = p_test_best
        final_th = best_pp['Thresh']
        
    ev_final_test = evaluate_events(y_test, p_test_final, final_th, merge_gap=1)
    fm_final_test = evaluate_frame_metrics(y_test, p_test_final)
    
    # 3. Target Uncoupled Circularity Test
    turnovers_unc = derive_turnovers_uncoupled(df_poss)
    df_lab_unc = label_turnover_horizons(df_poss, turnovers_unc)
    y_test_unc = df_lab_unc.iloc[split_idx:]['target_press_trigger'].values
    
    ev_unc_test = evaluate_events(y_test_unc, p_test_final, final_th, merge_gap=1)
    
    final_res = {
        'Baseline_Test': {**fm_base_test, **ev_base_test},
        'Candidate_Test': {**fm_final_test, **ev_final_test},
        'Uncoupled_Test': ev_unc_test,
        'Config': {
            'Features': best_cols,
            'Smoothing': best_pp['Smoothing'],
            'Threshold': best_pp['Thresh'],
            'HistGB_Params': best_hgb_cfg,
            'Ensemble': use_ensemble
        }
    }
    
    with open(f"{out_dir}/MODEL_IMPROVEMENT_RESULTS.json", "w") as f:
        json.dump(final_res, f, indent=4)
        
    pd.DataFrame([
        {'Model': 'Baseline', 'ROC': fm_base_test['roc'], 'Ev_Prec': ev_base_test['prec'], 'Ev_Rec': ev_base_test['rec'], 'Ev_F1': ev_base_test['f1']},
        {'Model': 'Candidate', 'ROC': fm_final_test['roc'], 'Ev_Prec': ev_final_test['prec'], 'Ev_Rec': ev_final_test['rec'], 'Ev_F1': ev_final_test['f1']}
    ]).to_csv(f"{out_dir}/FINAL_MODEL_COMPARISON.csv", index=False)
    
    print("Done!")

if __name__ == '__main__':
    main()
