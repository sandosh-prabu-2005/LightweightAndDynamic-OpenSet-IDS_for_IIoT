"""
Utility functions for data preprocessing, EVT, metrics, and normalization.

This module contains:
- Data loading and preprocessing
- Feature selection
- Dataset splitting (open-set)
- EVT (Extreme Value Theory) functions
- Metrics computation
- Normalization helpers
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import genpareto

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

SEED = 42
LABEL_UNKNOWN = 'Unknown'


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_load(path, name):
    """Safely load CSV file with error handling."""
    if not os.path.exists(path):
        print(f'  ❌ {name}: NOT FOUND → {path}')
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f'  ✅ {name:20s}: {df.shape[0]:>9,} rows × {df.shape[1]:>3} cols')
    return df


def preprocess(df, label_col, dataset_name='', verbose=True):
    """
    Preprocess dataset: clean, encode, and scale.
    
    Args:
        df: Input DataFrame
        label_col: Column name for labels
        dataset_name: Name for logging
        verbose: Print progress
    
    Returns:
        (X, y, scaler): Features, labels, fitted scaler
    """
    if verbose:
        print(f'\n🔹 {dataset_name}')
    
    df = df.copy()
    df.columns = df.columns.str.strip()
    before = len(df)
    df = df.drop_duplicates().fillna(0)
    
    if verbose:
        print(f'  Shape after clean: {df.shape}  (removed {before-len(df)} rows)')
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    
    y_raw = df[label_col].astype(str).values
    
    # Encode categorical columns
    cat_cols = [c for c in df.columns
                if c != label_col and not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(columns=[label_col]).apply(
        pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    if verbose:
        unique, counts = np.unique(y_raw, return_counts=True)
        print(f'  Classes: {dict(zip(unique, counts))}')
    
    return X, y_raw, scaler


def balance_data(X, y, max_majority=10000, min_minority=300, seed=SEED, verbose=True):
    """
    Balance dataset using SMOTE and undersampling.
    
    Args:
        X: Features
        y: Labels
        max_majority: Max samples per majority class
        min_minority: Min samples per minority class
        seed: Random seed
        verbose: Print progress
    
    Returns:
        (X_balanced, y_balanced)
    """
    counts = Counter(y)
    if verbose:
        print(f'  Before balance: {dict(counts)}')
    
    # Undersampling for majority classes
    under_strategy = {cls: min(cnt, max_majority)
                      for cls, cnt in counts.items() if cnt > max_majority}
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
        X, y = rus.fit_resample(X, y)
    
    # SMOTE for minority classes
    counts2 = Counter(y)
    min_count = min(counts2.values())
    k = min(5, min_count - 1)
    
    if k >= 1:
        over_strategy = {cls: max(cnt, min_minority)
                         for cls, cnt in counts2.items() if cnt < min_minority}
        if over_strategy:
            try:
                smote = SMOTE(sampling_strategy=over_strategy,
                              random_state=seed, k_neighbors=k)
                X, y = smote.fit_resample(X, y)
            except Exception as e:
                if verbose:
                    print(f'  ⚠️  SMOTE skipped: {e}')
    
    if verbose:
        print(f'  After  balance: {dict(Counter(y))}')
    
    return X, y


def select_features_mi(X, y, k_features, dataset_name=''):
    """
    Select top-k features using mutual information.
    
    Args:
        X: Features
        y: Labels
        k_features: Number of features to select
        dataset_name: Name for logging
    
    Returns:
        (X_selected, selector): Reduced features and fitted selector
    """
    k = min(k_features, X.shape[1])
    if dataset_name:
        print(f'🔹 {dataset_name} → {X.shape[1]} features to {k}')
    
    sel = SelectKBest(mutual_info_classif, k=k)
    X_sel = sel.fit_transform(X, y)
    
    if dataset_name:
        top_score = sel.scores_[sel.get_support()].max()
        print(f'   ⭐ Top MI score: {top_score:.4f}')
    
    return X_sel, sel


def create_open_set_split(X, y, unknown_classes, test_size=0.30, seed=SEED,
                         balance_train=True, verbose=True):
    """
    Create open-set split: withheld unknown classes at test time.
    
    Args:
        X: Features
        y: Labels
        unknown_classes: Classes to treat as unknown
        test_size: Test set fraction
        seed: Random seed
        balance_train: Whether to balance training set
        verbose: Print progress
    
    Returns:
        (X_train, y_train, X_test, y_test, known_classes, unknown_classes)
    """
    X_tr, X_te, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    
    # Remove unknown classes from training
    known_mask = ~np.isin(y_train, unknown_classes)
    X_tr, y_train = X_tr[known_mask], y_train[known_mask]
    
    # Balance training set
    if balance_train:
        X_tr, y_train = balance_data(X_tr, y_train, verbose=verbose)
    
    # Mark unknowns in test set
    y_test = np.array([LABEL_UNKNOWN if l in unknown_classes else l for l in y_test])
    known_cls = np.unique(y_train)
    unk_count = np.sum(y_test == LABEL_UNKNOWN)
    
    if verbose:
        print(f'   Train: {len(X_tr):,}  Test: {len(X_te):,}  Unknown-in-test: {unk_count:,}')
    
    return X_tr, y_train, X_te, y_test, known_cls, unknown_classes


# ============================================================================
# EVT (Extreme Value Theory) and Normalization Functions
# ============================================================================

def minmax_normalize(errors, e_min=None, e_max=None):
    """
    Min-max normalize reconstruction errors to [0, 1].
    
    Args:
        errors: Error values
        e_min: Minimum value (computed from errors if None)
        e_max: Maximum value (computed from errors if None)
    
    Returns:
        (normalized_errors, e_min, e_max)
    """
    if e_min is None:
        e_min = errors.min()
    if e_max is None:
        e_max = errors.max()
    
    denom = e_max - e_min if e_max - e_min > 1e-12 else 1.0
    return (errors - e_min) / denom, e_min, e_max


def zscore_normalize(errors, mean=None, std=None):
    """
    Z-score normalize reconstruction errors.
    
    Args:
        errors: Error values
        mean: Mean (computed from errors if None)
        std: Standard deviation (computed from errors if None)
    
    Returns:
        (normalized_errors, mean, std)
    """
    if mean is None:
        mean = errors.mean()
    if std is None:
        std = errors.std() + 1e-12
    
    return (errors - mean) / std, mean, std


def mean_excess_function(errors, u_values):
    """
    Compute Mean Excess Function (MEF) for threshold candidates.
    
    MEF: e(u) = E[X − u | X > u]
    
    Args:
        errors: Error values
        u_values: Threshold candidates
    
    Returns:
        List of (u, e_u, n_exceedances)
    """
    results = []
    for u in u_values:
        exc = errors[errors > u] - u
        if len(exc) < 5:
            break
        results.append((u, exc.mean(), len(exc)))
    return results


def find_mef_threshold(train_errors_norm, q_start=0.75, q_end=0.98, n_points=40):
    """
    Estimate threshold u via MEF slope-stabilization criterion.
    
    Args:
        train_errors_norm: Normalized training errors
        q_start: Starting quantile
        q_end: Ending quantile
        n_points: Number of points in scan
    
    Returns:
        Estimated threshold u
    """
    u_vals = np.quantile(train_errors_norm,
                        np.linspace(q_start, q_end, n_points))
    mef_pts = mean_excess_function(train_errors_norm, u_vals)
    
    if len(mef_pts) < 4:
        return float(np.quantile(train_errors_norm, (q_start + q_end) / 2))
    
    u_arr = np.array([p[0] for p in mef_pts])
    e_arr = np.array([p[1] for p in mef_pts])
    slopes = np.diff(e_arr) / (np.diff(u_arr) + 1e-12)
    slope_var = np.array([np.var(slopes[max(0, i-3):i+1])
                          for i in range(len(slopes))])
    stable_idx = np.argmin(slope_var)
    return float(u_arr[stable_idx])


def detect_unknown_evt(test_errors, train_errors,
                       tail_pct=0.10, q_start=0.75, q_end=0.98,
                       norm_mode='minmax'):
    """
    Stage 2 EVT pipeline: normalize, fit GPD, find threshold, classify.
    
    Args:
        test_errors: Test reconstruction errors
        train_errors: Training reconstruction errors
        tail_pct: Percentage of tail for GPD fitting
        q_start: Starting quantile for MEF scan
        q_end: Ending quantile for MEF scan
        norm_mode: 'minmax' or 'zscore' normalization
    
    Returns:
        (binary_predictions, threshold_u, train_errors_norm, test_errors_norm)
    """
    # Normalize using training stats
    if norm_mode == 'zscore':
        train_norm, t_mean, t_std = zscore_normalize(train_errors)
        test_norm, _, _ = zscore_normalize(test_errors, mean=t_mean, std=t_std)
    else:  # minmax
        train_norm, e_min, e_max = minmax_normalize(train_errors)
        denom = e_max - e_min if e_max - e_min > 1e-12 else 1.0
        test_norm = (test_errors - e_min) / denom
        test_norm = np.clip(test_norm, 0, None)
    
    # Tail-only GPD fit
    tail_cutoff = np.quantile(train_norm, 1 - tail_pct)
    tail_exc = train_norm[train_norm > tail_cutoff] - tail_cutoff
    
    if len(tail_exc) >= 10:
        try:
            xi, _, sigma = genpareto.fit(tail_exc, floc=0)
        except Exception:
            xi, sigma = 0.0, tail_exc.std() + 1e-8
    else:
        xi, sigma = (0.0, tail_exc.std() + 1e-8) if len(tail_exc) else (0.0, 0.01)
    
    # MEF threshold
    u = find_mef_threshold(train_norm, q_start=q_start, q_end=q_end)
    
    # Classify
    preds = np.where(test_norm > u, LABEL_UNKNOWN, 'Known').astype(object)
    
    return preds, u, train_norm, test_norm


# ============================================================================
# Metrics and Evaluation
# ============================================================================

def compute_detection_metrics(y_true, y_pred, label_unknown='Unknown'):
    """
    Compute classification and unknown detection metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_unknown: Label for unknown class
    
    Returns:
        Dict with Accuracy, Precision, Recall, F1, Unknown Detection Rate
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Unknown Detection Rate
    unk_mask = (y_true == label_unknown)
    if unk_mask.sum() > 0:
        udr = (y_pred[unk_mask] == label_unknown).mean()
    else:
        udr = float('nan')
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Unknown Detection Rate': udr
    }


def predict_open_set(model, X_test, recon_threshold, conf_threshold=0.70,
                     le=None, train_errors=None, norm_mode='minmax',
                     mode='hybrid', label_unknown='Unknown', device='cpu'):
    """
    Make open-set predictions using reconstruction error and/or confidence.
    
    Args:
        model: VAE model
        X_test: Test features
        recon_threshold: Reconstruction error threshold
        conf_threshold: Confidence threshold
        le: LabelEncoder for classes
        train_errors: Training errors for normalization
        norm_mode: 'minmax' or 'zscore'
        mode: 'recon_only', 'confidence_only', or 'hybrid'
        label_unknown: Label for unknown class
        device: Device for computation
    
    Returns:
        (predictions, max_confidence, normalized_recon_errors)
    """
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        recon_out, mu, _, logits = model(X_t)
        probs = logits.cpu().numpy()
        recon_err = F.mse_loss(recon_out, X_t, reduction='none').mean(dim=1).cpu().numpy()
    
    # Normalize reconstruction errors
    if train_errors is not None:
        if norm_mode == 'zscore':
            t_mean, t_std = train_errors.mean(), train_errors.std() + 1e-12
            recon_err_norm = (recon_err - t_mean) / t_std
        else:
            t_min, t_max = train_errors.min(), train_errors.max()
            denom = t_max - t_min if t_max - t_min > 1e-12 else 1.0
            recon_err_norm = np.clip((recon_err - t_min) / denom, 0, None)
    else:
        recon_err_norm = recon_err
    
    max_conf = probs.max(axis=1)
    pred_class = probs.argmax(axis=1)
    
    class_names = le.classes_ if le is not None else np.arange(probs.shape[1]).astype(str)
    
    predictions = []
    for i in range(len(X_test)):
        if mode == 'recon_only':
            is_unknown = recon_err_norm[i] > recon_threshold
        elif mode == 'confidence_only':
            is_unknown = max_conf[i] < conf_threshold
        else:  # hybrid
            is_unknown = (recon_err_norm[i] > recon_threshold) or (max_conf[i] < conf_threshold)
        
        predictions.append(label_unknown if is_unknown else class_names[pred_class[i]])
    
    return np.array(predictions), max_conf, recon_err_norm


def extract_latent_z(encoder, X, device='cpu'):
    """Extract latent representation z from encoder."""
    encoder.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mu, logvar = encoder(X_tensor)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
    
    return z


def build_update_dataset(X_known, y_known, X_unknown,
                        y_new_label='new_unknown',
                        samples_per_class=500, seed=SEED):
    """
    Build balanced update dataset for knowledge distillation.
    
    Args:
        X_known: Known class features
        y_known: Known class labels
        X_unknown: Unknown features
        y_new_label: Label for new unknown class
        samples_per_class: Samples per class
        seed: Random seed
    
    Returns:
        (X_update, y_update)
    """
    np.random.seed(seed)
    
    X_list = []
    y_list = []
    
    # Balance known classes
    unique_classes = np.unique(y_known)
    for cls in unique_classes:
        cls_mask = (y_known == cls)
        X_cls = X_known[cls_mask]
        
        if len(X_cls) > samples_per_class:
            idx = np.random.choice(len(X_cls), samples_per_class, replace=False)
            X_cls = X_cls[idx]
        
        y_cls = np.array([cls] * len(X_cls))
        X_list.append(X_cls)
        y_list.append(y_cls)
    
    # Add unknown samples
    if len(X_unknown) > samples_per_class:
        idx = np.random.choice(len(X_unknown), samples_per_class, replace=False)
        X_unknown = X_unknown[idx]
    
    y_unknown = np.array([y_new_label] * len(X_unknown))
    X_list.append(X_unknown)
    y_list.append(y_unknown)
    
    # Merge
    X_final = np.vstack(X_list)
    y_final = np.concatenate(y_list)
    
    return X_final, y_final
