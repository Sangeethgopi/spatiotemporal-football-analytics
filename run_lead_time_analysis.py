import os, sys, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS
from src.model import PressTriggerClassifier

def get_events(arr):
    events = []
    in_event = False
    start = 0
    for i in range(len(arr)):
        if arr[i] == 1 and not in_event:
            in_event = True; start = i
        elif arr[i] == 0 and in_event:
            in_event = False; events.append((start, i))
    if in_event: events.append((start, len(arr)))
    return events

print('Loading dataset...')
paths = ensure_dataset_available('data', 'match_1')
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

print('Training frozen Combined model...')
clf = PressTriggerClassifier(model_type='hist_gb')
clf.feature_names = FEATURE_COLUMNS
clf.fit(df_train, y_train)
y_prob = clf.predict_proba(df_val)[:, 1]

y_pred = (y_prob >= 0.50).astype(int)
true_events = get_events(y_val)
pred_events = get_events(y_pred)

tol = 50
matched_gt = {}
for p_idx, (p_start, p_end) in enumerate(pred_events):
    for t_idx, (t_start, t_end) in enumerate(true_events):
        if p_start <= t_end + tol and p_end >= max(0, t_start - tol):
            if t_idx not in matched_gt: matched_gt[t_idx] = []
            matched_gt[t_idx].append(p_idx)

results = []
for t_idx, (t_start, t_end) in enumerate(true_events):
    turnover_frame_in_val = t_end
    
    if t_idx in matched_gt:
        p_idx_list = matched_gt[t_idx]
        first_pred_frame = min(pred_events[p][0] for p in p_idx_list)
        first_prob = y_prob[first_pred_frame]
        lead_time_s = (turnover_frame_in_val - first_pred_frame) / 25.0
        
        results.append({
            'gt_event_idx': t_idx,
            'gt_frames': f'[{t_start}, {t_end}]',
            'detected': 'Yes',
            'first_pred_frame': first_pred_frame,
            'prob': first_prob,
            'lead_time_s': lead_time_s,
            'cat_ge0': lead_time_s >= 0,
            'cat_ge1': lead_time_s >= 1,
            'cat_ge2': lead_time_s >= 2,
            'cat_ge3': lead_time_s >= 3,
            'cat_ge4': lead_time_s >= 4,
            'matched_preds': str(p_idx_list)
        })
    else:
        results.append({
            'gt_event_idx': t_idx,
            'gt_frames': f'[{t_start}, {t_end}]',
            'detected': 'No',
            'first_pred_frame': '-',
            'prob': '-',
            'lead_time_s': None,
            'cat_ge0': False,
            'cat_ge1': False,
            'cat_ge2': False,
            'cat_ge3': False,
            'cat_ge4': False,
            'matched_preds': '-'
        })

df_res = pd.DataFrame(results)

print('\n### Phase 1: Lead-Time Analysis (Event-by-Event)\n')
print('| GT Event | GT Frames | Detected | First Pred Frame | Prob at Detect | Lead Time (s) | >=0s | >=1s | >=2s | >=3s | >=4s | Matched Pred Events |')
print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
for _, r in df_res.iterrows():
    if r['detected'] == 'Yes':
        print(f"| {r['gt_event_idx']} | {r['gt_frames']} | {r['detected']} | {r['first_pred_frame']} | {r['prob']:.3f} | {r['lead_time_s']:.2f} | {r['cat_ge0']} | {r['cat_ge1']} | {r['cat_ge2']} | {r['cat_ge3']} | {r['cat_ge4']} | {r['matched_preds']} |")
    else:
        print(f"| {r['gt_event_idx']} | {r['gt_frames']} | {r['detected']} | - | - | - | - | - | - | - | - | - |")

detected_df = df_res[df_res['detected'] == 'Yes']
lead_times = detected_df['lead_time_s'].values
n_gt = len(df_res)
n_det = len(detected_df)
n_miss = n_gt - n_det

print('\n### Aggregates\n')
print(f'- **Number of GT events:** {n_gt}')
print(f'- **Detected events:** {n_det}')
print(f'- **Missed events:** {n_miss}')
if n_det > 0:
    print(f'- **Median lead time:** {np.median(lead_times):.2f} s')
    print(f'- **Mean lead time:** {np.mean(lead_times):.2f} s')
    print(f'- **IQR:** {np.percentile(lead_times, 75) - np.percentile(lead_times, 25):.2f} s')
    print(f'- **Min / Max:** {np.min(lead_times):.2f} s / {np.max(lead_times):.2f} s')
    print(f'- **Proportion >= 1s:** {np.mean(lead_times >= 1.0):.2f}')
    print(f'- **Proportion >= 2s:** {np.mean(lead_times >= 2.0):.2f}')
    print(f'- **Proportion >= 3s:** {np.mean(lead_times >= 3.0):.2f}')
    print(f'- **Proportion >= 4s:** {np.mean(lead_times >= 4.0):.2f}')
