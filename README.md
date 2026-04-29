# 🛡️ Lightweight & Dynamic Open-Set IDS for IIoT

A modular VAE-based Intrusion Detection System with custom optimizers and Extreme Value Theory (EVT) for detecting unknown attacks in industrial IoT environments.

## 📋 Project Overview

This project implements a **three-stage architecture** for detecting known and unknown network attacks:

1. **Stage 1 (VAE + Teacher):** Variational Autoencoder with classification teacher for learning compressed representations
2. **Stage 2 (EVT):** Extreme Value Theory with Generalized Pareto Distribution for unknown detection
3. **Stage 3 (KD):** Knowledge Distillation for efficient deployment

**Key Innovation:** ABAO-V2 optimizer and ABAO+ with multi-loss adaptive weighting.

## 📁 Project Structure

```
.
├── model.py              # Neural network architectures
├── optimizer.py          # ABAO V1, V2, ABAO+ custom optimizers
├── utils.py             # Data preprocessing, EVT, metrics
├── train.py             # Main training pipeline
└── README.md            # This file
```

### model.py
Contains model definitions:
- **Encoder:** Maps features to latent space (μ, log σ²)
- **Decoder:** Reconstructs features from latent codes
- **TeacherClassifier:** Multi-layer classifier for known classes
- **VAEWithTeacher:** Full VAE + Teacher combined model
- **StudentNet:** Student network for knowledge distillation
- **total_vae_loss:** Combined VAE loss function

### optimizer.py
Custom optimizer implementations:
- **ABAO_V1:** Original Adaptive Boundary Aware Optimizer (reference)
- **ABAO_V2:** Improved EMA-normalized ABAO (recommended)
- **ABAOPlus:** Stability-Aware Adaptive Optimizer with multi-loss weighting

**Key difference (V2 vs V1):**
- V1 W_t collapses to 0.5 minimum → 50% weaker than Adam
- V2 uses EMA-normalized ratios with W_min=1.0 → never underperforms Adam

### utils.py
Helper functions for:
- Data loading and preprocessing (`safe_load`, `preprocess`, `balance_data`)
- Feature selection (`select_features_mi`)
- Open-set dataset splitting (`create_open_set_split`)
- EVT functions (`detect_unknown_evt`, `find_mef_threshold`, normalization)
- Metrics computation (`compute_detection_metrics`)
- Knowledge distillation (`extract_latent_z`, `build_update_dataset`)

### train.py
Main training script with three stages:
- **train_stage1:** VAE + Teacher training with divergence detection
- **stage2_evt_detection:** Unknown attack detection using EVT
- **stage3_knowledge_distillation:** Student network training
- **main:** Complete pipeline for all datasets

## 🚀 Quick Start

### Installation

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn tqdm imbalanced-learn
```

Optional (for Lion optimizer comparison):
```bash
pip install lion-pytorch
```

### Running the Training Pipeline

```bash
python train.py
```

This will:
1. Load datasets from `DATA_DIR` (edit path in `train.py`)
2. Preprocess and balance data
3. Train VAE + Teacher for each dataset
4. Apply EVT for unknown detection
5. Run knowledge distillation
6. Save trained models to `./saved_models/`

## 📊 Datasets

The system is designed for 4 datasets. Update paths in `train.py`:

| Dataset | Label Column | Unknown Class |
|---|---|---|
| NSL-KDD | `class` | `u2r` |
| CICIDS2017 | `Class` | `DoS`, `PortScan` |
| Gas Pipeline | `result` | `6` |
| Water Storage | `result` | `1` |

### Dataset Configuration

Per-dataset hyperparameters in `DS_CONFIG`:

```python
DS_CONFIG = {
    'NSL-KDD': {
        'latent_dim': 32,
        'epochs': 100,
        'beta_kl': 0.8,
        'k_features': 20,
        'evt_tail_pct': 0.10,
        'evt_q_start': 0.75,
        'evt_q_end': 0.98,
        'kd_epochs': 30,
        'evt_norm_mode': 'minmax',
    },
    # ... more datasets
}
```

Edit these values to tune performance per dataset.

## 🏗️ Architecture Details

### Stage 1: VAE + Teacher

**VAE Loss:** `L = L_recon + β·L_KL + L_classify`

- **Encoder:** input → 64 → 32 → latent_dim (μ, log σ²)
- **Decoder:** latent_dim → 32 → 64 → input
- **Teacher:** latent → 128×4 → Softmax (known class probabilities)

**KL Warmup:** β ramped from 0 → β_kl over first 40 epochs

**Divergence Detection:**
- NaN/Inf loss detection
- Sudden spike detection (loss > 5× best)
- Monotonic rise detection (10 consecutive epochs)

### Stage 2: EVT Unknown Detection

1. **Normalize** reconstruction errors (MinMax or Z-score)
2. **Fit GPD** to top `tail_pct%` of training errors (tail-only)
3. **Find threshold u** via Mean Excess Function (MEF) scan
4. **Classify:** `error > u → Unknown`

**Thresholds (tunable per dataset):**
- `evt_tail_pct`: Percentage of errors for GPD fitting (0.10 to 0.35)
- `evt_q_start`, `evt_q_end`: Quantile range for MEF scan (0.40 to 0.98)
- `evt_norm_mode`: MinMax or Z-score normalization

### Stage 3: Knowledge Distillation

- **Student Network:** Latent → 64 → Softmax (smaller than teacher)
- **KD Temperature:** T=4.0
- **Loss:** `0.7·KD_loss + 0.3·CE_loss`
- **Masked KD:** Train only on known classes (exclude detected unknowns)

## 🔬 Custom Optimizers

### ABAO V2 (Recommended)

**Formula:**
```
W_t = 1.0 + α·max(0, L_cls/EMA_cls−1) + β·max(0, L_rec/EMA_rec−1) + γ·(1−Conf)^τ
```

**Key features:**
- EMA-smoothed loss ratios: responds to relative difficulty
- W_min=1.0: never weaker than Adam on easy samples
- Decoupled weight decay (AdamW-style)
- Boundary signal amplification via (1−Conf)^τ

**Usage in train_stage1:**
```python
optimizer = optim.Adam(model.parameters(), lr=lr)  # Or ABAO_V2
```

### ABAO+ (Stability-Aware)

Decomposes into three loss streams with per-loss weighting:

```python
w_td, w_kl, w_recon = optimizer.get_adaptive_weights(loss_td, loss_kl, loss_recon)
loss = w_td * L_td + w_kl * (β * L_kl) + w_recon * L_recon
```

**Benefits:**
- Prevents single loss from dominating gradient landscape
- EMA stabilization prevents weight thrashing
- Embedded gradient clipping for stability

## 📈 Performance Targets

| Dataset | UDR (%) | F1 Score | KD Accuracy |
|---|---|---|---|
| NSL-KDD | 85-95 | - | 0.65-0.80 |
| CICIDS2017 | 65-75 | - | 0.70-0.82 |
| Gas Pipeline | 85-92 | - | 0.70-0.80 |
| Water Storage | 50-70 | - | 0.75-0.85 |

**UDR:** Unknown Detection Rate (how many true unknowns correctly identified)

## 🎯 Hybrid Unknown Detection

Combines reconstruction error and softmax confidence:

```python
if recon_error_norm > T  OR  max_softmax < C:
    prediction = "Unknown"
else:
    prediction = predicted_class
```

Visit `utils.predict_open_set()` for the implementation.

## 📋 Key Functions

### Training
```python
from train import train_stage1, stage2_evt_detection, stage3_knowledge_distillation
```

### Data Loading
```python
from utils import preprocess, select_features_mi, create_open_set_split
```

### Metrics
```python
from utils import compute_detection_metrics, predict_open_set
```

### Models
```python
from model import VAEWithTeacher, StudentNet, total_vae_loss
```

### Optimizers
```python
from optimizer import ABAO_V2, ABAOPlus
```

## 📊 Output Files

After running `train.py`:
- `./saved_models/` — Trained model checkpoints
  - `{dataset}_teacher.pt` — Stage 1 VAE + Teacher weights
  - `{dataset}_student.pt` — Stage 3 Student network weights

## 🔧 Configuration Tips

### Improve Performance

1. **Increase latent_dim** if reconstruction quality is poor (state explosion)
2. **Adjust β_kl** to balance reconstruction vs regularization
   - Lower β → focus on reconstruction (better for unknown detection)
   - Higher β → stricter regularization
3. **Tune tail_pct** for EVT
   - Higher % → more tail data for GPD, but noisier
   - Lower % → cleaner tail, but fewer samples
4. **Try ABAO_V2** optimizer instead of Adam for better convergence

### Dataset-Specific Tuning

**Water Storage (challenging dataset):**
- Higher `latent_dim: 48` (more expressive)
- Lower `beta_kl: 0.3` (prioritize reconstruction)
- Longer training `epochs: 130`
- Z-score normalization instead of MinMax

## 📝 Notes

- All code is CPU/GPU compatible (auto-detection)
- Reproducible: random seeds set to `SEED=42`
- No jupyter dependencies required (pure PyTorch + scikit-learn)
- Modular design allows easy swapping of components

## 🚀 Future Improvements

- [ ] Multi-GPU training support
- [ ] Automated hyperparameter search
- [ ] Export to ONNX for edge deployment
- [ ] Ensemble methods combining multiple stages
- [ ] Online learning for concept drift

## 📚 References

- **VAE:** Kingma & Welling (2013) - Auto-Encoding Variational Bayes
- **EVT:** de Haan & Ferreira (2006) - Extreme Value Theory
- **ABAO:** Custom adaptive boundary-aware optimizer (proposed)
- **KD:** Hinton et al. (2015) - Distilling the Knowledge in Neural Networks

## 📄 License

This project is provided as-is for research and educational purposes.

## ✉️ Contact

For questions or issues, please refer to the primary research publication.

---

**Last Updated:** April 2026  
**Version:** V14 (ABAO+ with multi-loss weighting)
