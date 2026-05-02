"""
Main training script for VAE-based Intrusion Detection System (IDS).

This script:
1. Loads and preprocesses datasets
2. Trains VAE + Teacher classifier (Stage 1)
3. Detects unknown attacks using EVT (Stage 2)
4. Performs knowledge distillation (Stage 3)
"""

import os
import sys
import time
import warnings
import pickle
import copy
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import custom modules
from model import VAEWithTeacher, StudentNet
from optimizer import ABAO_V2
from utils import (
    set_seed, safe_load, preprocess, apply_mi_feature_selection,
    fit_transform_feature_scaler,
    create_open_set_split, detect_unknown_evt, extract_latent_z,
    build_update_dataset, compute_detection_metrics, predict_open_set,
    make_class_weight_tensor, LABEL_UNKNOWN, SEED
)

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Datasets configuration per dataset
DS_CONFIG = {
    'NSL-KDD': {
        'latent_dim': 32,
        'epochs': 500,
        'beta_kl': 0.8,
        'k_features': 30,
        'evt_tail_pct': 0.10,
        'evt_q_start': 0.75,
        'evt_q_end': 0.98,
        'kd_epochs': 30,
        'evt_norm_mode': 'minmax',
    },
    'CICIDS2017': {
        'latent_dim': 32,
        'epochs': 500,
        'beta_kl': 1.0,
        'k_features': 40,
        'evt_tail_pct': 0.10,
        'evt_q_start': 0.75,
        'evt_q_end': 0.98,
        'kd_epochs': 30,
        'evt_norm_mode': 'minmax',
    },
    'Gas Pipeline': {
        'latent_dim': 32,
        'epochs': 500,
        'beta_kl': 1.0,
        'k_features': 25,
        'evt_tail_pct': 0.10,
        'evt_q_start': 0.75,
        'evt_q_end': 0.98,
        'kd_epochs': 30,
        'evt_norm_mode': 'minmax',
    },
    'Water Storage': {
        'latent_dim': 48,
        'epochs': 700,
        'beta_kl': 0.3,
        'k_features': None,
        'evt_tail_pct': 0.35,
        'evt_q_start': 0.40,
        'evt_q_end': 0.80,
        'kd_epochs': 60,
        'evt_norm_mode': 'zscore',
    },
}

# Global defaults
BATCH_SIZE = 512
LR = 3e-4
RF_THRESHOLD = 0.70
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True
VALIDATION_SPLIT = 0.20
EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_MIN_DELTA = 1e-4

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print(f'🖥️ Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'🚀 GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.empty_cache()

set_seed(SEED)


def vae_loss_components(model, xb, yb, beta, class_weights=None):
    """Compute VAE losses as GPU tensors for custom optimizer support."""
    recon, mu, logvar, logits = model(xb)
    loss_rec = F.mse_loss(recon, xb, reduction='mean')
    loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_cls = F.cross_entropy(logits, yb, weight=class_weights)
    recon_error = F.mse_loss(recon, xb, reduction='none').mean(dim=1)
    loss = loss_rec + beta * loss_kl + loss_cls
    return loss, loss_rec, loss_kl, loss_cls, recon_error, mu, logits


def step_abao_v2_optimizer(optimizer, loss, loss_rec, loss_cls, logits,
                           grad_clip=1.0):
    """Step ABAO-V2 with its classification, reconstruction, and confidence signals."""
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group['params']],
        max_norm=grad_clip
    )

    confidence = torch.softmax(logits.detach(), dim=1).max(dim=1).values.mean().item()
    optimizer.step(
        loss_cls=loss_cls.detach().item(),
        loss_rec=loss_rec.detach().item(),
        conf=confidence
    )

    return loss.detach().item()


def make_train_val_loaders(X, y, batch_size, validation_split=VALIDATION_SPLIT):
    """Create stratified train/validation loaders for known-class Stage 1 data."""
    stratify = y if validation_split > 0 and np.min(np.bincount(y)) >= 2 else None
    if validation_split > 0 and len(np.unique(y)) > 1:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X,
            y,
            test_size=validation_split,
            random_state=SEED,
            stratify=stratify,
        )
    else:
        X_fit, y_fit = X, y
        X_val, y_val = X[:0], y[:0]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_fit, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = None
    if len(X_val) > 0:
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    return train_loader, val_loader, y_fit, y_val


@torch.no_grad()
def evaluate_stage1_epoch(model, loader, beta, class_weights=None):
    model.eval()
    sum_loss = sum_Lr = sum_LKL = sum_Lc = 0.0
    correct = total = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=DEVICE.type == 'cuda')
        yb = yb.to(DEVICE, non_blocking=DEVICE.type == 'cuda')

        loss, loss_rec, loss_kl, loss_cls, _, _, logits = vae_loss_components(
            model, xb, yb, beta, class_weights=class_weights
        )
        n = len(xb)
        sum_loss += loss.detach().item() * n
        sum_Lr += loss_rec.detach().item() * n
        sum_LKL += loss_kl.detach().item() * n
        sum_Lc += loss_cls.detach().item() * n
        correct += (logits.argmax(1) == yb).sum().item()
        total += n

    return {
        'loss': sum_loss / total,
        'Lr': sum_Lr / total,
        'LKL': sum_LKL / total,
        'Lc': sum_Lc / total,
        'acc': correct / total,
    }

# ============================================================================
# Stage 1: Train VAE + Teacher Classifier
# ============================================================================

def train_stage1(X_train, y_train, n_classes,
                 epochs=100, batch_size=512, lr=1e-3,
                 latent_dim=32, beta_kl=1.0, verbose_every=10,
                 spike_factor=5.0, patience=10, grad_clip=1.0,
                 weight_decay=WEIGHT_DECAY,
                 validation_split=VALIDATION_SPLIT,
                 early_stopping_patience=EARLY_STOPPING_PATIENCE,
                 early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
                 class_names=None):
    """
    Train VAE + Teacher Classifier with divergence detection.
    
    Args:
        X_train: Training features
        y_train: Training labels (integer encoded)
        n_classes: Number of known classes
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        latent_dim: Latent dimension size
        beta_kl: KL divergence weight (β-VAE)
        verbose_every: Print every N epochs
        spike_factor: Loss spike detection threshold
        patience: Retained for backward compatibility
        grad_clip: Gradient clipping norm
        validation_split: Fraction of known training data used for validation
        early_stopping_patience: Epochs without validation improvement before stop
        early_stopping_min_delta: Minimum validation-loss improvement to reset patience
    
    Returns:
        (trained_model, history)
    """
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'\n🚀 Stage 1: Training VAE + Teacher')
    print(f'   Latent dim: {latent_dim}, Max epochs: {epochs}, β-KL: {beta_kl}')
    print(f'   Early stopping: patience={early_stopping_patience}, '
          f'min_delta={early_stopping_min_delta}, val_split={validation_split}')
    
    input_dim = X_train.shape[1]
    model = VAEWithTeacher(input_dim, latent_dim, n_classes).to(DEVICE)
    optimizer = ABAO_V2(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('⚙️ Using Optimizer: ABAO-V2')
    
    loader, val_loader, y_fit, _ = make_train_val_loaders(
        X_train,
        y_train,
        batch_size,
        validation_split=validation_split,
    )
    class_weights = make_class_weight_tensor(
        y_fit,
        n_classes,
        DEVICE,
        class_names=class_names,
        verbose=True,
    )
    
    history = []
    best_train_loss = math.inf
    best_val_loss = math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss = sum_Lr = sum_LKL = sum_Lc = 0.0
        correct = total = 0
        
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=DEVICE.type == 'cuda')
            yb = yb.to(DEVICE, non_blocking=DEVICE.type == 'cuda')
            
            # KL warmup: linear ramp over first 40 epochs
            beta = beta_kl * min(1.0, epoch / 40)
            loss, loss_rec, loss_kl, loss_cls, recon_error, z, logits = vae_loss_components(
                model, xb, yb, beta, class_weights=class_weights)

            if not torch.isfinite(loss):
                print(f'\n  ⛔ DIVERGENCE at epoch {epoch}: NaN/Inf')
                if best_state:
                    model.load_state_dict(best_state)
                return model, history
            
            batch_loss = step_abao_v2_optimizer(
                optimizer,
                loss,
                loss_rec,
                loss_cls,
                logits,
                grad_clip=grad_clip
            )
            
            n = len(xb)
            sum_loss += batch_loss * n
            sum_Lr += loss_rec.detach().item() * n
            sum_LKL += loss_kl.detach().item() * n
            sum_Lc += loss_cls.detach().item() * n
            correct += (logits.argmax(1) == yb).sum().item()
            total += n
        
        train_metrics = {
            'loss': sum_loss / total,
            'Lr': sum_Lr / total,
            'LKL': sum_LKL / total,
            'Lc': sum_Lc / total,
            'acc': correct / total,
        }
        val_metrics = (
            evaluate_stage1_epoch(model, val_loader, beta, class_weights=class_weights)
            if val_loader is not None else train_metrics
        )
        best_train_loss = min(best_train_loss, train_metrics['loss'])
        
        history.append({
            'epoch': epoch,
            'loss': train_metrics['loss'],
            'acc': train_metrics['acc'],
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['acc'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['acc'],
            'Lr': train_metrics['Lr'],
            'LKL': train_metrics['LKL'],
            'Lc': train_metrics['Lc'],
            'val_Lr': val_metrics['Lr'],
            'val_LKL': val_metrics['LKL'],
            'val_Lc': val_metrics['Lc'],
            'early_stop_monitor': 'val_loss' if val_loader is not None else 'train_loss',
        })
        
        improved = val_metrics['loss'] < best_val_loss - early_stopping_min_delta
        if improved:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Divergence detection
        if math.isnan(train_metrics['loss']) or math.isinf(train_metrics['loss']):
            print(f'\n  ⛔ DIVERGENCE at epoch {epoch}: NaN/Inf')
            if best_state:
                model.load_state_dict(best_state)
            return model, history
        
        if train_metrics['loss'] > best_train_loss * spike_factor and epoch > 5:
            print(f'\n  ⛔ DIVERGENCE at epoch {epoch}: Loss spike')
            if best_state:
                model.load_state_dict(best_state)
            return model, history

        if epochs_without_improvement >= early_stopping_patience:
            print(f'\n  ⏹️ Early stopping at epoch {epoch}. '
                  f'Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}')
            if best_state:
                model.load_state_dict(best_state)
            return model, history
        
        if epoch % verbose_every == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d}/{epochs} | '
                  f'TrainLoss={train_metrics["loss"]:.4f} '
                  f'ValLoss={val_metrics["loss"]:.4f} '
                  f'TrainAcc={train_metrics["acc"]:.4f} '
                  f'ValAcc={val_metrics["acc"]:.4f}')
    
    if best_state:
        model.load_state_dict(best_state)
    print(f'  ✅ Training complete. Restored epoch {best_epoch} '
          f'with best val_loss={best_val_loss:.4f}')
    return model, history


# ============================================================================
# Stage 2: EVT-based Unknown Detection
# ============================================================================

def stage2_evt_detection(model, X_train, X_test, y_test,
                        tail_pct=0.10, q_start=0.75, q_end=0.98,
                        norm_mode='minmax'):
    """
    Stage 2: Apply EVT for unknown attack detection.
    
    Args:
        model: Trained VAE model
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        tail_pct: Tail percentage for GPD
        q_start: MEF scan start quantile
        q_end: MEF scan end quantile
        norm_mode: 'minmax' or 'zscore'
    
    Returns:
        Dict with predictions and metrics
    """
    print(f'\n⚙️  Stage 2: EVT Unknown Detection (tail={tail_pct}, norm={norm_mode})')
    
    model.eval()
    train_errors = model.reconstruction_error(X_train, device=DEVICE)
    test_errors = model.reconstruction_error(X_test, device=DEVICE)
    
    preds_evt, u_threshold, tr_norm, te_norm = detect_unknown_evt(
        test_errors, train_errors,
        tail_pct=tail_pct,
        q_start=q_start,
        q_end=q_end,
        norm_mode=norm_mode
    )
    
    metrics = compute_detection_metrics(y_test, preds_evt)
    print(f'  EVT Results: Acc={metrics["Accuracy"]:.4f} '
          f'F1={metrics["F1"]:.4f} UDR={metrics["Unknown Detection Rate"]:.4f}')
    
    return {
        'predictions': preds_evt,
        'threshold': u_threshold,
        'train_errors': train_errors,
        'test_errors': test_errors,
        'metrics': metrics,
    }


# ============================================================================
# Stage 3: Knowledge Distillation
# ============================================================================

def stage3_knowledge_distillation(model, X_train, y_train, X_detected_unknown,
                                  n_classes, kd_epochs=30):
    """
    Stage 3: Train student network via knowledge distillation.
    
    Args:
        model: Trained teacher VAE
        X_train: Training data (known classes)
        y_train: Training labels
        X_detected_unknown: Detected unknown samples
        n_classes: Number of known classes
        kd_epochs: KD training epochs
    
    Returns:
        Trained student network
    """
    print(f'\n📚 Stage 3: Knowledge Distillation ({kd_epochs} epochs)')

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    model.eval()
    
    # Build balanced update dataset
    X_upd, y_upd = build_update_dataset(
        X_train, y_train, X_detected_unknown,
        y_new_label='new_unknown',
        samples_per_class=800
    )
    
    le_student = LabelEncoder()
    y_upd_enc = le_student.fit_transform(y_upd)
    
    # Extract latent codes
    z = extract_latent_z(model.encoder, X_upd, device=DEVICE)
    
    # Create student network
    student = StudentNet(z.shape[1], len(le_student.classes_)).to(DEVICE)
    print(f'\n🚀 Training using ABAO-V2 optimizer')
    optimizer = ABAO_V2(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f'⚙️ Using Optimizer: ABAO-V2 (Stage 3)')
    
    y_upd_tensor = torch.tensor(y_upd_enc, dtype=torch.long).to(DEVICE)
    student_class_weights = make_class_weight_tensor(
        y_upd_enc,
        len(le_student.classes_),
        DEVICE,
        class_names=le_student.classes_,
        verbose=True,
    )
    ce_loss_fn = nn.CrossEntropyLoss(weight=student_class_weights)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    new_unk_enc = int(le_student.transform(['new_unknown'])[0])
    known_mask = y_upd_tensor != new_unk_enc
    known_student_indices = torch.tensor(
        le_student.transform(np.unique(y_train)),
        dtype=torch.long,
        device=DEVICE
    )
    
    # KD training
    KD_T = 4.0
    for ep in range(1, kd_epochs + 1):
        student.train()
        student_logits = student(z)
        
        loss_ce = ce_loss_fn(student_logits, y_upd_tensor)
        
        if known_mask.sum() > 0:
            with torch.no_grad():
                teacher_logits = model.classifier(z[known_mask])
            student_known = student_logits[known_mask][:, known_student_indices]

            min_classes = min(student_known.size(1), teacher_logits.size(1))
            student_known = student_known[:, :min_classes]
            teacher_logits = teacher_logits[:, :min_classes]

            teacher_soft = torch.softmax(teacher_logits / KD_T, dim=1)
            log_probs = torch.log_softmax(student_known / KD_T, dim=1)
            loss_kd = (KD_T**2) * kd_loss_fn(log_probs, teacher_soft)
        else:
            loss_kd = torch.tensor(0.0, device=DEVICE)
        
        loss = 0.7 * loss_kd + 0.3 * loss_ce
        confidence = torch.softmax(student_logits.detach(), dim=1).max(dim=1).values.mean().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(
            loss_cls=loss_ce.detach().item(),
            loss_rec=loss_kd.detach().item(),
            conf=confidence
        )
        
        if ep % 10 == 0:
            print(f'  KD Epoch {ep}/{kd_epochs} | Loss={loss:.4f}')
    
    print(f'  ✅ KD complete')
    return student


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training pipeline for all datasets."""
    
    print('='*70)
    print('  VAE-based Intrusion Detection System (IDS)')
    print('='*70)
    
    # Dataset paths (update as needed)
    DATA_DIR = '/home/sandosh-prabu/Desktop/DATASET/'
    
    dataset_paths = {
        'NSL-KDD': (
            os.path.join(DATA_DIR, 'NSLKDD/nsl-train.csv'),
            'class'
        ),
        'CICIDS2017': (
            os.path.join(DATA_DIR, 'CICIDS2017/cicids-train-new.csv'),
            'Class'
        ),
        'Gas Pipeline': (
            os.path.join(DATA_DIR, 'gas_pipeline.csv'),
            'result'
        ),
        'Water Storage': (
            os.path.join(DATA_DIR, 'water_storage_tank.csv'),
            'result'
        ),
    }
    
    unknown_classes = {
        'NSL-KDD': ['u2r'],
        'CICIDS2017': ['DoS', 'PortScan'],
        'Gas Pipeline': ['6'],
        'Water Storage': ['1'],
    }
    
    # Load and preprocess datasets
    print('\n📂 Loading datasets...')
    datasets = {}
    for ds_name, (path, label_col) in dataset_paths.items():
        df = safe_load(path, ds_name)
        if df is not None:
            feature_names = [col for col in df.columns if col != label_col]
            X, y, scaler = preprocess(df, label_col, ds_name, verbose=False, scale=False)
            datasets[ds_name] = (X, y, scaler, feature_names)
    
    print(f'\n✅ Loaded {len(datasets)} datasets')
    
    # Training results
    all_results = {}
    
    # Process each dataset
    for ds_name, (X, y, scaler, feature_names) in datasets.items():
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        print(f'\n{"#"*70}')
        print(f'  ► Dataset: {ds_name}')
        print(f'{"#"*70}')
        print('\n🚀 Training using ABAO-V2 optimizer')
        
        cfg = DS_CONFIG.get(ds_name, {})
        latent_dim = cfg.get('latent_dim', 32)
        epochs = cfg.get('epochs', 100)
        beta_kl = cfg.get('beta_kl', 1.0)
        k_features = cfg.get('k_features')
        tail_pct = cfg.get('evt_tail_pct', 0.10)
        q_start = cfg.get('evt_q_start', 0.75)
        q_end = cfg.get('evt_q_end', 0.98)
        kd_epochs = cfg.get('kd_epochs', 30)
        norm_mode = cfg.get('evt_norm_mode', 'minmax')
        ds_lr = 2e-4 if ds_name == 'Water Storage' else LR
        
        res = {}
        
        # Open-set split
        X_tr, y_tr, X_te, y_te, known_cls, unk_cls = create_open_set_split(
            X, y,
            unknown_classes.get(ds_name, []),
            test_size=0.30,
            balance_train=False,
            dataset_name=ds_name,
            verbose=True,
        )

        # Fit MI selector only on known training data; reuse columns for test data.
        X_tr, X_te, feature_selector = apply_mi_feature_selection(
            X_tr,
            y_tr,
            X_test=X_te,
            dataset_name=ds_name,
            k_features=k_features,
            feature_names=feature_names,
            seed=SEED,
            verbose=True,
        )
        res['feature_selector'] = feature_selector

        X_tr, X_te, feature_scaler = fit_transform_feature_scaler(
            X_tr,
            X_te,
            verbose=True,
        )
        res['feature_scaler'] = feature_scaler
        
        # Encode labels
        le = LabelEncoder()
        le.fit(known_cls)
        y_tr_enc = le.transform(y_tr)
        n_cls = len(le.classes_)
        
        # Stage 1: Train VAE + Teacher
        teacher, hist1 = train_stage1(
            X_tr, y_tr_enc, n_cls,
            epochs=epochs, batch_size=BATCH_SIZE,
            lr=ds_lr, latent_dim=latent_dim, beta_kl=beta_kl,
            weight_decay=WEIGHT_DECAY,
            class_names=le.classes_,
        )
        res['teacher'] = teacher
        res['history1'] = hist1
        
        # Stage 2: EVT Detection
        evt_res = stage2_evt_detection(
            teacher, X_tr, X_te, y_te,
            tail_pct=tail_pct, q_start=q_start, q_end=q_end,
            norm_mode=norm_mode
        )
        res['evt'] = evt_res
        
        # Stage 3: Knowledge Distillation (if unknowns detected)
        detected_unk_mask = (evt_res['predictions'] == LABEL_UNKNOWN)
        if detected_unk_mask.sum() > 0:
            X_unk_detected = X_te[detected_unk_mask]
            student = stage3_knowledge_distillation(
                teacher, X_tr, y_tr, X_unk_detected,
                n_cls, kd_epochs=kd_epochs
            )
            res['student'] = student
        
        all_results[ds_name] = res
    
    # Save models
    print(f'\n💾 Saving models...')
    os.makedirs('saved_models', exist_ok=True)
    for ds_name, res in all_results.items():
        safe_name = ds_name.replace(' ', '_')
        if 'teacher' in res:
            path = f'saved_models/{safe_name}_teacher.pt'
            torch.save(res['teacher'].state_dict(), path)
            print(f'  ✅ {path}')
        if 'feature_selector' in res:
            selector_path = f'saved_models/{safe_name}_feature_selector.pkl'
            with open(selector_path, 'wb') as f:
                pickle.dump(res['feature_selector'], f)
            print(f'  ✅ {selector_path}')
        if 'feature_scaler' in res:
            scaler_path = f'saved_models/{safe_name}_feature_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(res['feature_scaler'], f)
            print(f'  ✅ {scaler_path}')
    
    print('\n✅ Training complete!')
    return all_results


if __name__ == '__main__':
    results = main()
