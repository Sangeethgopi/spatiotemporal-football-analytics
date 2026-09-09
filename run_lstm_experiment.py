import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, brier_score_loss, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from src.download_metrica import ensure_dataset_available
from src.data_loader import load_match_tracking
from src.possession import assign_frame_possession, derive_turnover_events, label_turnover_horizons
from src.features import extract_pressing_features, FEATURE_COLUMNS

# Fix seed
torch.manual_seed(42)
np.random.seed(42)

def evaluate_events_detection(y_true, y_prob, threshold, tolerance_frames=50):
    y_pred = (y_prob >= threshold).astype(int)
    def get_events(arr):
        events = []
        in_event = False
        start_idx = 0
        for i in range(len(arr)):
            if arr[i] == 1 and not in_event:
                in_event = True
                start_idx = i
            elif arr[i] == 0 and in_event:
                in_event = False
                events.append((start_idx, i))
        if in_event:
            events.append((start_idx, len(arr)))
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

def calculate_lead_time(y_true, y_prob, threshold, tolerance_frames=50):
    y_pred = (y_prob >= threshold).astype(int)
    def get_events(arr):
        events = []
        in_event = False
        start_idx = 0
        for i in range(len(arr)):
            if arr[i] == 1 and not in_event:
                in_event = True
                start_idx = i
            elif arr[i] == 0 and in_event:
                in_event = False
                events.append((start_idx, i))
        if in_event:
            events.append((start_idx, len(arr)))
        return events
        
    true_events = get_events(y_true)
    lead_times = []
    
    for t_start, t_end in true_events:
        # Ground truth turnover frame is exactly t_end
        turnover_frame = t_end
        
        # Search for FIRST prediction >= 0.5 within [t_start, t_end]
        block = y_prob[t_start:t_end]
        if np.any(block >= threshold):
            first_idx_in_block = np.argmax(block >= threshold)
            first_crossing_frame = t_start + first_idx_in_block
            lead_frames = turnover_frame - first_crossing_frame
            lead_s = lead_frames / 25.0
            lead_times.append(lead_s)
            
    if len(lead_times) > 0:
        return {
            'detected_events': len(lead_times),
            'median_lead_time': float(np.median(lead_times)),
            'mean_lead_time': float(np.mean(lead_times)),
            'iqr': float(np.percentile(lead_times, 75) - np.percentile(lead_times, 25)),
            'min': float(np.min(lead_times)),
            'max': float(np.max(lead_times))
        }
    return None

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take the last timestep
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out # Return logits!

def create_sequences(X, y, seq_length):
    # X shape: (N, features), y shape: (N,)
    N = len(X)
    X_seq = np.zeros((N, seq_length, X.shape[1]), dtype=np.float32)
    y_seq = np.zeros((N,), dtype=np.float32)
    
    # Fill sequences. 
    # For i < seq_length-1, pad with the first frame (or zeros, but padding with first frame is safer).
    for i in range(N):
        if i < seq_length - 1:
            pad_len = seq_length - 1 - i
            seq = np.vstack([np.tile(X[0], (pad_len, 1)), X[:i+1]])
        else:
            seq = X[i-seq_length+1 : i+1]
        X_seq[i] = seq
        y_seq[i] = y[i]
        
    return torch.tensor(X_seq), torch.tensor(y_seq).view(-1, 1)

def run_experiment():
    os.makedirs('data/results/temporal_lstm', exist_ok=True)
    
    print('Loading data...')
    paths = ensure_dataset_available('data', 'match_1')
    df_raw = load_match_tracking(paths['home'], paths['away'])
    df_poss = assign_frame_possession(df_raw)
    turnovers_df = derive_turnover_events(df_poss)
    df_labeled = label_turnover_horizons(df_poss, turnovers_df)
    df_features = extract_pressing_features(df_labeled)
    
    # Fill NaNs with 0 as in original setup
    df_features.fillna(0, inplace=True)

    n_frames = len(df_features)
    split_idx = int(n_frames * 0.70)
    
    # Strict chronological holdout.
    df_train_full = df_features.iloc[:split_idx - 100].copy()
    df_test = df_features.iloc[split_idx:].copy()
    
    # Validation split from training ONLY (first 80% train, last 20% val)
    train_n = len(df_train_full)
    val_split_idx = int(train_n * 0.8)
    df_train = df_train_full.iloc[:val_split_idx].copy()
    df_val = df_train_full.iloc[val_split_idx:].copy()

    X_train = df_train[FEATURE_COLUMNS].values
    y_train = df_train['target_press_trigger'].values
    
    X_val = df_val[FEATURE_COLUMNS].values
    y_val = df_val['target_press_trigger'].values
    
    X_test = df_test[FEATURE_COLUMNS].values
    y_test = df_test['target_press_trigger'].values
    
    # Scale based on train ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Class weight calculation on training only
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    print(f'Positive class weight: {pos_weight.item()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    windows = [25, 50, 75]
    results = {}
    lead_times = {}

    for w in windows:
        print(f'\n--- Training LSTM with sequence length: {w} ---')
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, w)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, w)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, w)
        
        train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=256, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_seq, y_test_seq), batch_size=256, shuffle=False)

        model = SimpleLSTM(input_dim=len(FEATURE_COLUMNS)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        patience = 5
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(50):
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = model(X_b)
                    loss = criterion(out, y_b)
                    val_loss += loss.item()
                    
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print('Early stopping.')
                break
                
        model.load_state_dict(best_model_state)
        
        # Test evaluation
        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in test_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        
        y_prob = np.concatenate(preds).flatten()
        
        # Metrics
        res, pred_ev, true_ev, mat_t, tp_p = evaluate_events_detection(y_test, y_prob, 0.50, 50)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        res['roc_auc'] = roc_auc_score(y_test, y_prob)
        res['pr_auc'] = auc(rec, prec)
        res['brier'] = brier_score_loss(y_test, y_prob)
        
        results[f'LSTM_{w}'] = res
        lead_time = calculate_lead_time(y_test, y_prob, 0.50, 50)
        lead_times[f'LSTM_{w}'] = lead_time
        print(f"Results window={w}: ROC={res['roc_auc']:.3f}, EventF1={res['event_f1']:.3f}")
        
        # Save primary 50-frame probabilities for error analysis
        if w == 50:
            df_test['lstm_prob'] = y_prob
            df_test.to_csv('data/results/temporal_lstm/test_preds_50.csv', index=False)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))
            res['confusion_matrix'] = cm.tolist()
            
            # Extract Error Analysis indices
            errors = {'TP': [], 'FP': [], 'FN': []}
            for p_idx in list(tp_p)[:5]:
                start, end = pred_ev[p_idx]
                errors['TP'].append({'frames': [start, end]})
            fp_idx = [i for i in range(len(pred_ev)) if i not in tp_p]
            for p_idx in fp_idx[:5]:
                start, end = pred_ev[p_idx]
                errors['FP'].append({'frames': [start, end]})
            fn_idx = [i for i in range(len(true_ev)) if i not in mat_t]
            for t_idx in fn_idx[:5]:
                start, end = true_ev[t_idx]
                errors['FN'].append({'frames': [start, end]})
            
            with open('data/results/temporal_lstm/LSTM_ERROR_ANALYSIS_RAW.json', 'w') as f:
                json.dump(errors, f, indent=2)

    # Load baseline
    with open('data/results/rigorous_eval/AUTHORITATIVE_RESULTS.json', 'r') as f:
        auth = json.load(f)
    baseline = auth['primary']
    baseline_lead = auth.get('lead_time', {'median_lead_time': 1.16, 'mean_lead_time': 1.25}) # Provide placeholder if missing, but we computed it before. Actually I'll just write it down in the final CSV.
    
    # Save results
    with open('data/results/temporal_lstm/LSTM_RESULTS.json', 'w') as f:
        json.dump({'LSTM': results, 'LeadTimes': lead_times}, f, indent=2)
        
    config = {
        'model': 'LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dropout(0.2) -> Dense(1, sigmoid)',
        'batch_size': 256,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'patience': 5,
        'features': FEATURE_COLUMNS,
        'windows_tested': windows,
        'scaler': 'StandardScaler fitted on train_split',
        'class_weight': pos_weight.item(),
        'val_split': 'Last 20% of the 70% chronological train block'
    }
    with open('data/results/temporal_lstm/LSTM_CONFIGURATION.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    # Write comparison CSV
    lstm50 = results['LSTM_50']
    df_comp = pd.DataFrame({
        'Metric': ['ROC-AUC', 'PR-AUC', 'Frame Precision', 'Frame Recall', 'Frame F1', 'Event Precision', 'Event Recall', 'Event F1', 'Median Lead Time (s)'],
        'HistGB': [baseline.get('roc_auc', 0), baseline.get('pr_auc', 0), baseline['frame_precision'], baseline['frame_recall'], baseline['frame_f1'], baseline['event_precision'], baseline['event_recall'], baseline['event_f1'], 1.16],
        'LSTM_50': [lstm50['roc_auc'], lstm50['pr_auc'], lstm50['frame_precision'], lstm50['frame_recall'], lstm50['frame_f1'], lstm50['event_precision'], lstm50['event_recall'], lstm50['event_f1'], lead_times['LSTM_50']['median_lead_time'] if lead_times['LSTM_50'] else 0.0]
    })
    df_comp.to_csv('data/results/temporal_lstm/LSTM_MODEL_COMPARISON.csv', index=False)
    
    df_sens = pd.DataFrame({
        'Window': [25, 50, 75],
        'ROC-AUC': [results['LSTM_25']['roc_auc'], results['LSTM_50']['roc_auc'], results['LSTM_75']['roc_auc']],
        'PR-AUC': [results['LSTM_25']['pr_auc'], results['LSTM_50']['pr_auc'], results['LSTM_75']['pr_auc']],
        'Event F1': [results['LSTM_25']['event_f1'], results['LSTM_50']['event_f1'], results['LSTM_75']['event_f1']],
        'Event Recall': [results['LSTM_25']['event_recall'], results['LSTM_50']['event_recall'], results['LSTM_75']['event_recall']],
        'Median Lead Time': [lead_times['LSTM_25']['median_lead_time'] if lead_times['LSTM_25'] else 0.0, 
                             lead_times['LSTM_50']['median_lead_time'] if lead_times['LSTM_50'] else 0.0, 
                             lead_times['LSTM_75']['median_lead_time'] if lead_times['LSTM_75'] else 0.0]
    })
    df_sens.to_csv('data/results/temporal_lstm/LSTM_WINDOW_SENSITIVITY.csv', index=False)
    
if __name__ == '__main__':
    run_experiment()
