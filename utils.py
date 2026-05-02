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
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

SEED = 42
LABEL_UNKNOWN = 'Unknown'

DATASET_FEATURE_COUNTS = {
    'NSL-KDD': 30,
    'CICIDS2017': 40,
    'Gas Pipeline': 25,
    'Water Storage': None,
}


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


def preprocess(df, label_col, dataset_name='', verbose=True, scale=True):
    """
    Preprocess dataset: clean, encode, and scale.
    
    Args:
        df: Input DataFrame
        label_col: Column name for labels
        dataset_name: Name for logging
        verbose: Print progress
        scale: Whether to fit StandardScaler immediately. For research
            train/test pipelines, pass False and fit scaling on training data.
    
    Returns:
        (X, y, scaler): Features, labels, fitted scaler
    """
    if verbose:
        print(f'\n🔹 {dataset_name}')
    
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    keep_duplicates = dataset_name.strip().lower() == 'water storage'
    if keep_duplicates:
        df = df.fillna(0)
    else:
        df = df.drop_duplicates().fillna(0)
    
    if verbose:
        if keep_duplicates:
            print(f'  Shape after clean: {df.shape}  (duplicates kept for Water Storage)')
        else:
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
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    
    if verbose:
        unique, counts = np.unique(y_raw, return_counts=True)
        print(f'  Classes: {dict(zip(unique, counts))}')
    
    return X, y_raw, scaler


def fit_transform_feature_scaler(X_train, X_test=None, verbose=True):
    """Fit StandardScaler on training features and reuse it for test features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    if verbose:
        print(f'   Scaler fit on train only: {X_train.shape[1]} features')
    if X_test is None:
        return X_train_scaled, scaler
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    return X_train_scaled, X_test_scaled, scaler


def _format_distribution(y, max_items=12):
    counts = Counter(y)
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    shown = ', '.join(f'{cls}:{cnt}' for cls, cnt in items[:max_items])
    if len(items) > max_items:
        shown += f', ... (+{len(items) - max_items} classes)'
    return '{' + shown + '}'


def log_class_distribution(title, y, max_items=12):
    print(f'   {title}: {_format_distribution(y, max_items=max_items)}')


def _stratify_labels_or_none(y):
    _, counts = np.unique(y, return_counts=True)
    if len(counts) > 1 and counts.min() >= 2:
        return y
    return None


def balance_data(
    X,
    y,
    max_majority=None,
    min_minority=0,
    seed=SEED,
    verbose=True,
    majority_keep_fraction=0.50,
    majority_multiplier=8,
    min_majority_keep=50000,
):
    """
    Moderately balance data without collapsing large classes.

    This function is intentionally conservative. The research pipeline now
    prefers class weighting, and this sampler is kept for explicit hybrid
    experiments only.
    
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
        log_class_distribution('Before balancing', y)
    
    nonzero_counts = np.array(list(counts.values()), dtype=np.int64)
    median_count = int(np.median(nonzero_counts)) if len(nonzero_counts) else 0
    adaptive_cap = max(min_majority_keep, int(median_count * majority_multiplier))
    if max_majority is not None:
        adaptive_cap = max(int(max_majority), min_majority_keep)

    under_strategy = {}
    for cls, cnt in counts.items():
        keep_by_fraction = int(np.ceil(cnt * majority_keep_fraction))
        target = max(keep_by_fraction, adaptive_cap)
        target = min(cnt, target)
        if target < cnt:
            under_strategy[cls] = target

    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)
        X, y = rus.fit_resample(X, y)
    
    # Optional minority support for experiments. Default is disabled because
    # class weights preserve the original sample distribution more faithfully.
    counts2 = Counter(y)
    min_count = min(counts2.values()) if counts2 else 0
    k = min(5, min_count - 1)

    if min_minority > 0 and k >= 1:
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
        log_class_distribution('After balancing ', y)
    
    return X, y


def compute_class_weight_vector(y_encoded, n_classes=None, class_names=None, verbose=True):
    """Compute sklearn balanced class weights for encoded training labels."""
    y_encoded = np.asarray(y_encoded, dtype=np.int64)
    if n_classes is None:
        n_classes = int(y_encoded.max()) + 1 if y_encoded.size else 0

    weights = np.ones(n_classes, dtype=np.float32)
    present_classes = np.unique(y_encoded)
    if present_classes.size:
        present_weights = compute_class_weight(
            class_weight='balanced',
            classes=present_classes,
            y=y_encoded,
        ).astype(np.float32)
        weights[present_classes] = present_weights

    if verbose:
        names = (
            list(class_names)
            if class_names is not None and len(class_names) == n_classes
            else [str(i) for i in range(n_classes)]
        )
        summary = ', '.join(
            f'{names[idx]}={weights[idx]:.4f}' for idx in range(n_classes)
        )
        print(f'   Class weights: {summary}')

    return weights


def make_class_weight_tensor(y_encoded, n_classes, device, class_names=None, verbose=True):
    weights = compute_class_weight_vector(
        y_encoded,
        n_classes=n_classes,
        class_names=class_names,
        verbose=verbose,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


@dataclass
class MIFeatureSelector:
    """Small reusable MI selector that stores and reapplies selected columns."""
    selected_indices: np.ndarray
    selected_names: list
    feature_names: list
    mi_scores: np.ndarray
    k_features: int
    original_feature_count: int
    dataset_name: str = ''
    reduction_applied: bool = True

    def transform(self, X):
        return X[:, self.selected_indices].astype(np.float32, copy=False)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices
        mask = np.zeros(self.original_feature_count, dtype=bool)
        mask[self.selected_indices] = True
        return mask


def dataset_feature_count(dataset_name, override=None):
    """Resolve dataset-specific MI feature count; None means use all features."""
    if override is not None:
        return override
    return DATASET_FEATURE_COUNTS.get(dataset_name, 20)


def _default_feature_names(n_features):
    return [f'feature_{idx}' for idx in range(n_features)]


def _log_mi_selection(selector, top_n=10):
    if not selector.dataset_name:
        return

    action = (
        f'{selector.original_feature_count} features to {selector.k_features}'
        if selector.reduction_applied
        else f'using all {selector.original_feature_count} features'
    )
    print(f'🔹 {selector.dataset_name} → {action}')

    if selector.mi_scores.size == 0:
        print('   MI ranking skipped because no feature reduction was applied.')
        return

    ranked = np.argsort(selector.mi_scores)[::-1]
    print('   Top MI scores:')
    for rank, idx in enumerate(ranked[:top_n], start=1):
        name = selector.feature_names[idx]
        print(f'     {rank:2d}. {name}: {selector.mi_scores[idx]:.6f}')

    selected = ', '.join(selector.selected_names[:top_n])
    if len(selector.selected_names) > top_n:
        selected += ', ...'
    print(f'   Selected {len(selector.selected_names)} features: {selected}')


def fit_mi_feature_selector(
    X_train,
    y_train,
    dataset_name='',
    k_features=None,
    feature_names=None,
    seed=SEED,
    verbose=True,
    max_mi_samples=200000,
):
    """
    Fit a Mutual Information feature selector on training data only.

    Args:
        X_train: Training features
        y_train: Training labels
        dataset_name: Dataset name used to resolve default feature count
        k_features: Optional override; None may mean no reduction for configured datasets
        feature_names: Optional original feature names
        seed: Random seed for mutual_info_classif
        verbose: Print selected feature details
        max_mi_samples: Stratified sample size for MI scoring on very large data

    Returns:
        MIFeatureSelector with stored feature indices/names and MI scores
    """
    n_features = X_train.shape[1]
    feature_names = list(feature_names) if feature_names is not None else _default_feature_names(n_features)
    if len(feature_names) != n_features:
        feature_names = _default_feature_names(n_features)

    resolved_k = dataset_feature_count(dataset_name, override=k_features)
    if resolved_k is None or resolved_k >= n_features:
        selector = MIFeatureSelector(
            selected_indices=np.arange(n_features),
            selected_names=feature_names,
            feature_names=feature_names,
            mi_scores=np.array([], dtype=np.float32),
            k_features=n_features,
            original_feature_count=n_features,
            dataset_name=dataset_name,
            reduction_applied=False,
        )
        if verbose:
            _log_mi_selection(selector)
        return selector

    k = max(1, min(int(resolved_k), n_features))
    X_mi, y_mi = X_train, y_train
    if max_mi_samples is not None and len(X_train) > max_mi_samples:
        indices = np.arange(len(X_train))
        stratify = _stratify_labels_or_none(y_train)
        sample_idx, _ = train_test_split(
            indices,
            train_size=int(max_mi_samples),
            random_state=seed,
            stratify=stratify,
        )
        X_mi, y_mi = X_train[sample_idx], y_train[sample_idx]
        if verbose and dataset_name:
            print(f'   MI ranking sample: {len(X_mi):,}/{len(X_train):,} training rows')

    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=seed)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
    selected_indices = np.argsort(mi_scores)[::-1][:k]
    selected_indices = np.sort(selected_indices)
    selected_names = [feature_names[idx] for idx in selected_indices]

    selector = MIFeatureSelector(
        selected_indices=selected_indices,
        selected_names=selected_names,
        feature_names=feature_names,
        mi_scores=mi_scores,
        k_features=k,
        original_feature_count=n_features,
        dataset_name=dataset_name,
        reduction_applied=True,
    )
    if verbose:
        _log_mi_selection(selector)
    return selector


def apply_mi_feature_selection(
    X_train,
    y_train,
    X_test=None,
    dataset_name='',
    k_features=None,
    feature_names=None,
    seed=SEED,
    verbose=True,
    max_mi_samples=200000,
):
    """Fit MI selector on training data, then reuse it for train/test transforms."""
    selector = fit_mi_feature_selector(
        X_train,
        y_train,
        dataset_name=dataset_name,
        k_features=k_features,
        feature_names=feature_names,
        seed=seed,
        verbose=verbose,
        max_mi_samples=max_mi_samples,
    )
    X_train_selected = selector.transform(X_train)
    if X_test is None:
        return X_train_selected, selector
    return X_train_selected, selector.transform(X_test), selector


def select_features_mi(
    X,
    y,
    k_features=None,
    dataset_name='',
    feature_names=None,
    seed=SEED,
    max_mi_samples=200000,
):
    """
    Backward-compatible helper for selecting features from a single matrix.

    Prefer apply_mi_feature_selection for train/test pipelines so the selector is
    fitted on training data only and then reused for validation/test data.
    
    Args:
        X: Features
        y: Labels
        k_features: Number of features to select
        dataset_name: Name for logging
    
    Returns:
        (X_selected, selector): Reduced features and fitted selector
    """
    return apply_mi_feature_selection(
        X,
        y,
        X_test=None,
        dataset_name=dataset_name,
        k_features=k_features,
        feature_names=feature_names,
        seed=seed,
        verbose=True,
        max_mi_samples=max_mi_samples,
    )


def create_open_set_split(
    X,
    y,
    unknown_classes,
    test_size=0.30,
    seed=SEED,
    balance_train=False,
    verbose=True,
    dataset_name='',
    balance_kwargs=None,
):
    """
    Create open-set split: withheld unknown classes at test time.
    
    Args:
        X: Features
        y: Labels
        unknown_classes: Classes to treat as unknown
        test_size: Test set fraction
        seed: Random seed
        balance_train: Whether to apply optional moderate hybrid balancing
        verbose: Print progress
    
    Returns:
        (X_train, y_train, X_test, y_test, known_classes, unknown_classes)
    """
    test_size = 0.30 if test_size is None else test_size
    stratify = _stratify_labels_or_none(y)
    X_tr_all, X_te, y_train_all, y_test_raw = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    
    # Remove unknown classes from training
    known_mask = ~np.isin(y_train_all, unknown_classes)
    X_tr, y_train = X_tr_all[known_mask], y_train_all[known_mask]

    if verbose:
        title = f'{dataset_name} split' if dataset_name else 'Open-set split'
        split_mode = 'stratified' if stratify is not None else 'non-stratified fallback for singleton classes'
        print(f'\n🔀 {title}: 70/30 {split_mode} train/test')
        log_class_distribution('Full distribution', y)
        log_class_distribution('Train before unknown removal', y_train_all)
        removed_unknown = y_train_all[~known_mask]
        if len(removed_unknown):
            log_class_distribution('Unknown removed from train', removed_unknown)
        log_class_distribution('Known train before balancing', y_train)
        log_class_distribution('Raw test distribution', y_test_raw)
    
    if balance_train:
        kwargs = balance_kwargs or {}
        X_tr, y_train = balance_data(X_tr, y_train, verbose=verbose, **kwargs)
    elif verbose:
        print('   Balancing: disabled; class weights should handle imbalance.')
    
    # Mark unknowns in test set
    y_test = np.array([LABEL_UNKNOWN if l in unknown_classes else l for l in y_test_raw])
    known_cls = np.unique(y_train)
    unk_count = np.sum(y_test == LABEL_UNKNOWN)
    
    if verbose:
        log_class_distribution('Final known train distribution', y_train)
        log_class_distribution('Final open-set test distribution', y_test)
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
        probs = torch.softmax(logits, dim=1).cpu().numpy()
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
