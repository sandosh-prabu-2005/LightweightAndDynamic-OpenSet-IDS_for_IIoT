"""
Generate dataset analysis plots for the modular open-set IDS project.

Outputs:
  plots/<DATASET>_before.png
  plots/<DATASET>_after.png
  plots/<DATASET>_confusion.png

Run:
  python analysis_visualization.py

Optional:
  DATA_DIR=/path/to/DATASET python analysis_visualization.py
"""

import os
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from model import VAEWithTeacher
from utils import (
    LABEL_UNKNOWN,
    SEED,
    create_open_set_split,
    preprocess,
    safe_load,
    select_features_mi,
    set_seed,
)


warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR = "plots"
SAVED_MODELS_DIR = "saved_models"


DS_CONFIG = {
    "NSL-KDD": {
        "latent_dim": 32,
        "k_features": 20,
        "path_parts": ("NSLKDD", "nsl-train.csv"),
        "label_col": "class",
        "unknown_classes": ["u2r"],
        "model_file": "NSL-KDD_teacher.pt",
    },
    "CICIDS2017": {
        "latent_dim": 32,
        "k_features": 20,
        "path_parts": ("CICIDS2017", "cicids-train-new.csv"),
        "label_col": "Class",
        "unknown_classes": ["DoS", "PortScan"],
        "model_file": "CICIDS2017_teacher.pt",
    },
    "Gas Pipeline": {
        "latent_dim": 32,
        "k_features": 20,
        "path_parts": ("gas_pipeline.csv",),
        "label_col": "result",
        "unknown_classes": ["6"],
        "model_file": "Gas_Pipeline_teacher.pt",
    },
    "Water Storage": {
        "latent_dim": 48,
        "k_features": 23,
        "path_parts": ("water_storage_tank.csv",),
        "label_col": "result",
        "unknown_classes": ["1"],
        "model_file": "Water_Storage_teacher.pt",
    },
}


def sanitize_filename(name):
    """Use the same readable dataset names requested by the plotting spec."""
    return name.replace(" ", "_")


def dataset_path(data_dir, cfg):
    return os.path.join(data_dir, *cfg["path_parts"])


def plot_class_distribution(labels, title, output_path):
    """Plot class counts as a bar chart."""
    counts_by_class = Counter(labels)
    class_names = [str(cls) for cls in counts_by_class.keys()]
    counts = list(counts_by_class.values())

    width = max(8, min(24, 0.45 * max(1, len(class_names))))
    plt.figure(figsize=(width, 6))
    plt.bar(class_names, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, dataset_name, output_path):
    """Plot and save confusion matrix with stable known/unknown label ordering."""
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    if LABEL_UNKNOWN in labels:
        labels = [label for label in labels if label != LABEL_UNKNOWN] + [LABEL_UNKNOWN]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    width = max(8, min(28, 0.65 * max(1, len(labels))))
    height = max(6, min(24, 0.55 * max(1, len(labels))))
    plt.figure(figsize=(width, height))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_checkpoint_state(path):
    """Load either a raw state_dict or a checkpoint dict containing model_state_dict."""
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def infer_model_shape(state_dict):
    """Infer model dimensions from saved weights when possible."""
    input_dim = state_dict["encoder.fc1.weight"].shape[1]
    latent_dim = state_dict["encoder.fc_mu.weight"].shape[0]
    n_classes = state_dict["classifier.net.8.weight"].shape[0]
    return int(input_dim), int(latent_dim), int(n_classes)


def load_model(path, input_dim=None, latent_dim=None, n_classes=None):
    """Create VAEWithTeacher, load saved weights, and switch to eval mode."""
    state_dict = load_checkpoint_state(path)
    ckpt_input_dim, ckpt_latent_dim, ckpt_n_classes = infer_model_shape(state_dict)

    input_dim = ckpt_input_dim if input_dim is None else input_dim
    latent_dim = ckpt_latent_dim if latent_dim is None else latent_dim
    n_classes = ckpt_n_classes if n_classes is None else n_classes

    if (input_dim, latent_dim, n_classes) != (
        ckpt_input_dim,
        ckpt_latent_dim,
        ckpt_n_classes,
    ):
        raise ValueError(
            "Model shape mismatch for "
            f"{path}: requested (input_dim={input_dim}, latent_dim={latent_dim}, "
            f"n_classes={n_classes}) but checkpoint has "
            f"(input_dim={ckpt_input_dim}, latent_dim={ckpt_latent_dim}, "
            f"n_classes={ckpt_n_classes})."
        )

    model = VAEWithTeacher(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_classes=n_classes,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_known_classes(model, X_test, label_encoder, batch_size=4096):
    """Predict known-class labels from model classifier output."""
    preds = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            xb = torch.as_tensor(
                X_test[start : start + batch_size],
                dtype=torch.float32,
                device=DEVICE,
            )
            _, _, _, logits = model(xb)
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.append(batch_preds)

    pred_indices = np.concatenate(preds) if preds else np.array([], dtype=int)
    valid_mask = pred_indices < len(label_encoder.classes_)
    if not np.all(valid_mask):
        pred_labels = np.full(len(pred_indices), LABEL_UNKNOWN, dtype=object)
        pred_labels[valid_mask] = label_encoder.inverse_transform(
            pred_indices[valid_mask]
        )
        return pred_labels.astype(str)

    return label_encoder.inverse_transform(pred_indices).astype(str)


def process_dataset(dataset_name, cfg, data_dir):
    """Create all requested plots for one dataset."""
    path = dataset_path(data_dir, cfg)
    label_col = cfg["label_col"]
    safe_name = sanitize_filename(dataset_name)

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    df = safe_load(path, dataset_name)
    if df is None:
        print(f"Skipping {dataset_name}: dataset file not found.")
        return

    df.columns = df.columns.str.strip()
    if label_col not in df.columns:
        raise ValueError(
            f"{dataset_name}: label column '{label_col}' not found in {path}. "
            f"Available columns include: {list(df.columns[:10])}"
        )

    raw_labels = df[label_col].astype(str).values
    before_path = os.path.join(PLOTS_DIR, f"{safe_name}_before.png")
    plot_class_distribution(
        raw_labels,
        "Class Distribution (Before Preprocessing)",
        before_path,
    )
    print(f"Saved: {before_path}")

    X, y, _ = preprocess(df, label_col, dataset_name, verbose=False)

    model_path = os.path.join(SAVED_MODELS_DIR, cfg["model_file"])
    if not os.path.exists(model_path):
        print(f"Skipping model plots for {dataset_name}: missing {model_path}")
        return

    state_dict = load_checkpoint_state(model_path)
    model_input_dim, model_latent_dim, model_n_classes = infer_model_shape(state_dict)
    k_features = model_input_dim if model_input_dim <= X.shape[1] else cfg["k_features"]

    X_selected, _ = select_features_mi(X, y, k_features, dataset_name)
    X_train, y_train, X_test, y_test, known_classes, _ = create_open_set_split(
        X_selected,
        y,
        cfg["unknown_classes"],
        test_size=0.30,
        seed=SEED,
        balance_train=True,
        verbose=False,
    )

    after_path = os.path.join(PLOTS_DIR, f"{safe_name}_after.png")
    plot_class_distribution(
        y_train,
        "Class Distribution (After Preprocessing)",
        after_path,
    )
    print(f"Saved: {after_path}")

    label_encoder = LabelEncoder()
    label_encoder.fit(known_classes)

    if len(label_encoder.classes_) != model_n_classes:
        raise ValueError(
            f"{dataset_name}: LabelEncoder has {len(label_encoder.classes_)} "
            f"known classes, but checkpoint classifier has {model_n_classes} outputs. "
            "Check unknown_classes and the preprocessing/split settings."
        )

    model = load_model(
        model_path,
        input_dim=model_input_dim,
        latent_dim=model_latent_dim,
        n_classes=model_n_classes,
    )

    if X_test.shape[1] != model_input_dim:
        raise ValueError(
            f"{dataset_name}: X_test has {X_test.shape[1]} features, but model "
            f"expects {model_input_dim}. Check k_features/model checkpoint."
        )

    y_true = np.asarray(y_test).astype(str)
    y_pred_labels = predict_known_classes(model, X_test, label_encoder)

    confusion_path = os.path.join(PLOTS_DIR, f"{safe_name}_confusion.png")
    plot_confusion_matrix(y_true, y_pred_labels, dataset_name, confusion_path)
    print(f"Saved: {confusion_path}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data_dir = os.environ.get("DATA_DIR", "/home/sandosh-prabu/Desktop/DATASET/")
    print(f"DATA_DIR: {data_dir}")

    for dataset_name, cfg in DS_CONFIG.items():
        process_dataset(dataset_name, cfg, data_dir)

    print(f"\nDone. Plots saved under: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
