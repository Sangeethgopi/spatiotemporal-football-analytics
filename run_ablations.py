import os, sys
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd
import numpy as np
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from src.model import PressTriggerClassifier
from src.evaluate import evaluate_predictions, evaluate_events
from src.baseline_ppda import RollingPPDABaselineClassifier, DistanceBaselineClassifier

def run():
    print('Loading data...')
    paths = ensure_dataset_available(data_dir='data', match_id='match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)
    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    df_features = extract_pressing_features(df_labeled)
    
    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    
    df_train = df_features.iloc[:split_idx - 100].copy()
    df_val = df_features.iloc[split_idx:].copy()
    
    y_train = df_train['target_press_trigger'].values
    y_val = df_val['target_press_trigger'].values
    
    spatial_cols = ['feat_def_dist_1', 'feat_def_dist_2', 'feat_def_dist_3', 'feat_def_dist_mean3', 'feat_def_density_5m', 'feat_def_density_10m', 'feat_att_density_10m', 'feat_numerical_advantage_10m', 'feat_passing_lane_occlusion', 'feat_viable_passing_options', 'feat_dist_to_goal', 'feat_dist_to_sideline', 'feat_ball_x', 'feat_ball_y']
    kinematic_cols = ['feat_closing_speed_max', 'feat_closing_speed_mean3', 'feat_ball_speed', 'feat_ball_progression_x', 'feat_roll_closing_speed_mean']
    shape_cols = ['feat_def_hull_area', 'feat_def_block_width', 'feat_def_block_depth', 'feat_def_dispersion', 'feat_def_centroid_dist_to_ball', 'feat_def_line_height', 'feat_roll_hull_area_change']
    combined_cols = FEATURE_COLUMNS
    
    models = {
        'Combined (HistGB)': ('hist_gb', combined_cols),
        'Spatial-Only (HistGB)': ('hist_gb', spatial_cols),
        'Kinematic-Only (HistGB)': ('hist_gb', kinematic_cols),
        'Shape-Only (HistGB)': ('hist_gb', shape_cols),
        'Combined (Logistic)': ('logistic', combined_cols)
    }
    
    results = []
    
    for name, (m_type, cols) in models.items():
        print(f'Training {name}...')
        clf = PressTriggerClassifier(model_type=m_type)
        clf.feature_names = cols
        clf.fit(df_train, y_train)
        probs = clf.predict_proba(df_val)[:, 1]
        
        frm_res = evaluate_predictions(y_val, probs)
        evt_res = evaluate_events(y_val, probs)
        
        results.append({
            'Model': name,
            'ROC-AUC': round(frm_res['roc_auc'], 3),
            'Frm Prec': round(frm_res['precision'], 3),
            'Frm Rec': round(frm_res['recall'], 3),
            'Evt Prec': round(evt_res['event_precision'], 3),
            'Evt Rec': round(evt_res['event_recall'], 3),
            'Evt F1': round(evt_res['event_f1'], 3),
            'TP Evts': evt_res['event_tp'],
            'FP Evts': evt_res['event_fp']
        })
        
    print('Training Baselines...')
    b_ppda = RollingPPDABaselineClassifier(window_sec=15.0)
    b_ppda.fit(df_train, y_train)
    p_ppda = b_ppda.predict_proba(df_val)[:, 1]
    frm_res = evaluate_predictions(y_val, p_ppda)
    evt_res = evaluate_events(y_val, p_ppda)
    results.append({'Model': 'Baseline (PPDA)', 'ROC-AUC': round(frm_res['roc_auc'], 3), 'Frm Prec': round(frm_res['precision'], 3), 'Frm Rec': round(frm_res['recall'], 3), 'Evt Prec': round(evt_res['event_precision'], 3), 'Evt Rec': round(evt_res['event_recall'], 3), 'Evt F1': round(evt_res['event_f1'], 3), 'TP Evts': evt_res['event_tp'], 'FP Evts': evt_res['event_fp']})
    
    b_dist = DistanceBaselineClassifier(dist_threshold=3.0)
    p_dist = b_dist.predict_proba(df_val)[:, 1]
    frm_res = evaluate_predictions(y_val, p_dist)
    evt_res = evaluate_events(y_val, p_dist)
    results.append({'Model': 'Baseline (Distance 3m)', 'ROC-AUC': round(frm_res['roc_auc'], 3), 'Frm Prec': round(frm_res['precision'], 3), 'Frm Rec': round(frm_res['recall'], 3), 'Evt Prec': round(evt_res['event_precision'], 3), 'Evt Rec': round(evt_res['event_recall'], 3), 'Evt F1': round(evt_res['event_f1'], 3), 'TP Evts': evt_res['event_tp'], 'FP Evts': evt_res['event_fp']})

    df_res = pd.DataFrame(results)
    print("\n--- ABLATION RESULTS ---")
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    run()
