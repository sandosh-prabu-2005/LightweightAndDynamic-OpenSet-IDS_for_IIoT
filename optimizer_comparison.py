import argparse
import copy
import math
import os
import random
import warnings
from dataclasses import dataclass

# NOTE: Final training uses ABAO-V2 based on multi-dataset evaluation.
# This script is experimental and kept for research-only optimizer comparisons.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import VAEWithTeacher
from optimizer import ABAO_V2, CRAZYFOX
from utils import (
    LABEL_UNKNOWN,
    SEED,
    compute_detection_metrics,
    create_open_set_split,
    detect_unknown_evt,
    preprocess,
    safe_load,
    select_features_mi,
    set_seed,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATHS = {
    "NSL-KDD": ("NSLKDD/nsl-train.csv", "class"),
    "CICIDS2017": ("CICIDS2017/cicids-train-new.csv", "Class"),
    "Gas Pipeline": ("gas_pipeline.csv", "result"),
    "Water Storage": ("water_storage_tank.csv", "result"),
}

UNKNOWN_CLASSES = {
    "NSL-KDD": ["u2r"],
    "CICIDS2017": ["DoS", "PortScan"],
    "Gas Pipeline": ["6"],
    "Water Storage": ["1"],
}

DATASET_CONFIG = {
    "NSL-KDD": {
        "latent_dim": 32,
        "k_features": 20,
        "beta_kl": 0.8,
        "tail_pct": 0.10,
        "q_start": 0.75,
        "q_end": 0.98,
        "norm_mode": "minmax",
    },
    "CICIDS2017": {
        "latent_dim": 32,
        "k_features": 20,
        "beta_kl": 1.0,
        "tail_pct": 0.10,
        "q_start": 0.75,
        "q_end": 0.98,
        "norm_mode": "minmax",
    },
    "Gas Pipeline": {
        "latent_dim": 32,
        "k_features": 20,
        "beta_kl": 1.0,
        "tail_pct": 0.10,
        "q_start": 0.75,
        "q_end": 0.98,
        "norm_mode": "minmax",
    },
    "Water Storage": {
        "latent_dim": 48,
        "k_features": 23,
        "beta_kl": 0.3,
        "tail_pct": 0.35,
        "q_start": 0.40,
        "q_end": 0.80,
        "norm_mode": "zscore",
    },
}


@dataclass
class PreparedDataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    y_train_enc: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    n_classes: int
    input_dim: int


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_path(data_dir, dataset_name):
    relative_path, label_col = DATASET_PATHS[dataset_name]
    return os.path.join(data_dir, relative_path), label_col


def prepare_dataset(args):
    cfg = DATASET_CONFIG[args.dataset]
    dataset_path, label_col = resolve_dataset_path(args.data_dir, args.dataset)
    df = safe_load(dataset_path, args.dataset)
    if df is None:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X, y, _ = preprocess(df, label_col, args.dataset, verbose=True)
    k_features = args.k_features if args.k_features is not None else cfg["k_features"]
    X, _ = select_features_mi(X, y, k_features, args.dataset)

    X_train, y_train, X_test, y_test, known_classes, _ = create_open_set_split(
        X,
        y,
        UNKNOWN_CLASSES.get(args.dataset, []),
        test_size=args.test_size,
        seed=args.seed,
        balance_train=True,
        verbose=True,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(known_classes)
    y_train_enc = label_encoder.transform(y_train)

    return PreparedDataset(
        name=args.dataset,
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        y_train_enc=y_train_enc.astype(np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        label_encoder=label_encoder,
        n_classes=len(label_encoder.classes_),
        input_dim=X_train.shape[1],
    )


def available_optimizers():
    names = ["Adam", "AdamW", "SGD", "RMSprop"]
    if hasattr(optim, "NAdam"):
        names.append("Nadam")
    names.extend(["ABAO-V2", "CRAZYFOX"])
    return names


def make_optimizer(optimizer_name, parameters, lr, weight_decay):
    parameters = list(parameters)
    if optimizer_name == "Adam":
        return optim.Adam(parameters, lr=lr)
    if optimizer_name == "AdamW":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "SGD":
        return optim.SGD(parameters, lr=lr, momentum=0.9)
    if optimizer_name == "RMSprop":
        return optim.RMSprop(parameters, lr=lr, momentum=0.9)
    if optimizer_name == "Nadam":
        return optim.NAdam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "ABAO-V2":
        return ABAO_V2(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "CRAZYFOX":
        return CRAZYFOX(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def loss_components(model, xb, yb, beta_kl):
    recon, mu, logvar, logits = model(xb)
    recon_error = F.mse_loss(recon, xb, reduction="none").mean(dim=1)
    loss_recon = F.mse_loss(recon, xb, reduction="mean")
    loss_kl = beta_kl * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_cls = F.cross_entropy(logits, yb)
    loss = loss_recon + loss_kl + loss_cls
    return loss, loss_recon, loss_kl, loss_cls, logits, recon_error, mu


def step_optimizer(
    optimizer,
    optimizer_name,
    loss,
    loss_recon,
    loss_kl,
    loss_cls,
    logits,
    target=None,
    epoch=None,
    total_epochs=None,
):
    optimizer.zero_grad()

    if hasattr(optimizer, "get_adaptive_weights"):
        w_cls, w_kl, w_recon = optimizer.get_adaptive_weights(
            loss_cls.detach().item(),
            loss_kl.detach().item(),
            loss_recon.detach().item(),
        )
        loss = w_cls * loss_cls + w_kl * loss_kl + w_recon * loss_recon

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]],
        max_norm=1.0,
    )

    if optimizer_name == "ABAO-V2":
        confidence = logits.detach().max(dim=1).values.mean().item()
        optimizer.step(
            loss_cls=loss_cls.detach().item(),
            loss_rec=loss_recon.detach().item(),
            conf=confidence,
        )
    elif optimizer_name == "CRAZYFOX":
        confidence = logits.detach().max(dim=1).values.mean().item()
        optimizer.step(
            loss_cls=loss_cls.detach().item(),
            loss_kl=loss_kl.detach().item(),
            loss_rec=loss_recon.detach().item(),
            conf=confidence,
            logits=logits.detach(),
            target=target.detach() if target is not None else None,
            epoch=epoch,
            total_epochs=total_epochs,
        )
    else:
        optimizer.step()

    return loss.detach().item()


def train_model(
    prepared,
    optimizer_name,
    epochs,
    batch_size,
    lr,
    latent_dim,
    beta_kl,
    weight_decay,
    seed,
):
    set_all_seeds(seed)

    model = VAEWithTeacher(
        input_dim=prepared.input_dim,
        latent_dim=latent_dim,
        n_classes=prepared.n_classes,
    ).to(DEVICE)

    optimizer = make_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)

    X_tensor = torch.tensor(prepared.X_train, dtype=torch.float32)
    y_tensor = torch.tensor(prepared.y_train_enc, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        beta_eff = beta_kl * min(1.0, epoch / 40.0)

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            loss, loss_recon, loss_kl, loss_cls, logits, recon_error, z = loss_components(
                model, xb, yb, beta_eff
            )

            if optimizer_name == "CRAZYFOX":
                loss = optimizer.compute_loss(
                    loss_cls,
                    loss_kl,
                    loss_recon,
                    recon_error=recon_error,
                    z=z,
                    target=yb,
                    epoch=epoch,
                    total_epochs=epochs,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{optimizer_name} produced a non-finite loss at epoch {epoch}"
                )

            batch_loss = step_optimizer(
                optimizer,
                optimizer_name,
                loss,
                loss_recon,
                loss_kl,
                loss_cls,
                logits,
                yb,
                epoch=epoch,
                total_epochs=epochs,
            )

            batch_size_actual = xb.size(0)
            total_loss += batch_loss * batch_size_actual
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += batch_size_actual

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_seen,
                "train_accuracy": total_correct / total_seen,
            }
        )

    return model, history


@torch.no_grad()
def batched_reconstruction_errors_and_predictions(model, X, label_encoder, batch_size):
    model.eval()
    errors = []
    labels = []

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    for (xb,) in loader:
        xb = xb.to(DEVICE)
        recon, _, _, logits = model(xb)
        batch_errors = F.mse_loss(recon, xb, reduction="none").mean(dim=1)
        batch_pred_idx = logits.argmax(dim=1).detach().cpu().numpy()

        errors.append(batch_errors.detach().cpu().numpy())
        labels.extend(label_encoder.classes_[batch_pred_idx])

    return np.concatenate(errors), np.array(labels, dtype=object)


def evaluate_model(prepared, model, batch_size, tail_pct, q_start, q_end, norm_mode):
    train_errors, _ = batched_reconstruction_errors_and_predictions(
        model,
        prepared.X_train,
        prepared.label_encoder,
        batch_size,
    )
    test_errors, known_predictions = batched_reconstruction_errors_and_predictions(
        model,
        prepared.X_test,
        prepared.label_encoder,
        batch_size,
    )

    evt_predictions, threshold, train_norm, test_norm = detect_unknown_evt(
        test_errors,
        train_errors,
        tail_pct=tail_pct,
        q_start=q_start,
        q_end=q_end,
        norm_mode=norm_mode,
    )

    final_predictions = np.where(
        evt_predictions == LABEL_UNKNOWN,
        LABEL_UNKNOWN,
        known_predictions,
    ).astype(object)

    metrics = compute_detection_metrics(
        prepared.y_test,
        final_predictions,
        label_unknown=LABEL_UNKNOWN,
    )

    y_unknown = (prepared.y_test == LABEL_UNKNOWN).astype(int)
    if len(np.unique(y_unknown)) == 2:
        auroc = roc_auc_score(y_unknown, test_norm)
    else:
        auroc = float("nan")

    return {
        "accuracy": metrics["Accuracy"],
        "precision": metrics["Precision"],
        "recall": metrics["Recall"],
        "f1": metrics["F1"],
        "auroc": auroc,
        "udr": metrics["Unknown Detection Rate"],
        "evt_threshold": threshold,
        "train_recon_error_mean": float(np.mean(train_errors)),
        "test_recon_error_mean": float(np.mean(test_errors)),
    }


def train_and_evaluate(
    optimizer_name,
    prepared,
    epochs,
    batch_size,
    lr,
    latent_dim,
    beta_kl,
    weight_decay,
    tail_pct,
    q_start,
    q_end,
    norm_mode,
    seed,
):
    model, history = train_model(
        prepared=prepared,
        optimizer_name=optimizer_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        latent_dim=latent_dim,
        beta_kl=beta_kl,
        weight_decay=weight_decay,
        seed=seed,
    )
    metrics = evaluate_model(
        prepared=prepared,
        model=model,
        batch_size=batch_size,
        tail_pct=tail_pct,
        q_start=q_start,
        q_end=q_end,
        norm_mode=norm_mode,
    )
    metrics["optimizer"] = optimizer_name
    metrics["final_train_loss"] = history[-1]["loss"] if history else math.nan
    return metrics, model, history


def plot_results(df, output_path, title="Optimizer Comparison (F1 Score)"):
    plt.figure(figsize=(10, 5))
    plt.bar(df["optimizer"], df["f1"])
    plt.title(title)
    plt.xlabel("Optimizer")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0.0, max(1.0, float(df["f1"].max()) * 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare optimizers for VAE-based IDS training."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_PATHS.keys(),
        default="NSL-KDD",
        help="Retained for compatibility; this script runs all datasets.",
    )
    parser.add_argument("--data-dir", default="/home/sandosh-prabu/Desktop/DATASET")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--k-features", type=int, default=None)
    parser.add_argument("--beta-kl", type=float, default=None)
    parser.add_argument("--tail-pct", type=float, default=None)
    parser.add_argument("--q-start", type=float, default=None)
    parser.add_argument("--q-end", type=float, default=None)
    parser.add_argument("--norm-mode", choices=["minmax", "zscore"], default=None)
    parser.add_argument("--csv-path", default="optimizer_per_dataset.csv")
    parser.add_argument("--average-csv-path", default="optimizer_average.csv")
    parser.add_argument("--plot-path", default="avg_optimizer_f1.png")
    parser.add_argument("--model-path", default="best_optimizer_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Datasets: {', '.join(DATASET_PATHS.keys())}")
    print(f"Comparison epochs: {args.epochs}")

    all_results = []
    failed = []

    for dataset_name in DATASET_PATHS.keys():
        print(f"\n{'=' * 72}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 72}")

        args.dataset = dataset_name
        cfg = copy.deepcopy(DATASET_CONFIG[dataset_name])
        latent_dim = args.latent_dim if args.latent_dim is not None else cfg["latent_dim"]
        beta_kl = args.beta_kl if args.beta_kl is not None else cfg["beta_kl"]
        tail_pct = args.tail_pct if args.tail_pct is not None else cfg["tail_pct"]
        q_start = args.q_start if args.q_start is not None else cfg["q_start"]
        q_end = args.q_end if args.q_end is not None else cfg["q_end"]
        norm_mode = args.norm_mode if args.norm_mode is not None else cfg["norm_mode"]

        try:
            prepared = prepare_dataset(args)
        except Exception as exc:
            print(f"Dataset preparation failed for {dataset_name}: {exc}")
            for optimizer_name in available_optimizers():
                failed.append((dataset_name, optimizer_name, str(exc)))
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "optimizer": optimizer_name,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "auroc": float("nan"),
                        "udr": float("nan"),
                        "evt_threshold": float("nan"),
                        "train_recon_error_mean": float("nan"),
                        "test_recon_error_mean": float("nan"),
                        "final_train_loss": float("nan"),
                    }
                )
            continue

        for optimizer_name in available_optimizers():
            print(f"\nTraining {optimizer_name} on {dataset_name}")
            try:
                metrics, _, _ = train_and_evaluate(
                    optimizer_name=optimizer_name,
                    prepared=prepared,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    latent_dim=latent_dim,
                    beta_kl=beta_kl,
                    weight_decay=args.weight_decay,
                    tail_pct=tail_pct,
                    q_start=q_start,
                    q_end=q_end,
                    norm_mode=norm_mode,
                    seed=args.seed,
                )
                metrics["dataset"] = dataset_name
                all_results.append(metrics)
            except Exception as exc:
                failed.append((dataset_name, optimizer_name, str(exc)))
                all_results.append(
                    {
                        "dataset": dataset_name,
                        "optimizer": optimizer_name,
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "auroc": float("nan"),
                        "udr": float("nan"),
                        "evt_threshold": float("nan"),
                        "train_recon_error_mean": float("nan"),
                        "test_recon_error_mean": float("nan"),
                        "final_train_loss": float("nan"),
                    }
                )

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        by=["dataset", "f1"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    avg_df = df.groupby("optimizer").mean(numeric_only=True).reset_index()
    avg_df = avg_df.sort_values(by="f1", ascending=False, kind="mergesort").reset_index(drop=True)

    df.to_csv(args.csv_path, index=False)
    avg_df.to_csv(args.average_csv_path, index=False)
    plot_results(avg_df, args.plot_path, title="Average F1 Across Datasets")

    display_cols = ["dataset", "optimizer", "accuracy", "precision", "recall", "f1", "auroc", "udr"]
    per_dataset_display = df[display_cols].rename(
        columns={
            "dataset": "Dataset",
            "optimizer": "Optimizer",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auroc": "AUROC",
            "udr": "UDR",
        }
    )

    avg_display = avg_df[["optimizer", "accuracy", "precision", "recall", "f1", "auroc", "udr"]].rename(
        columns={
            "optimizer": "Optimizer",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auroc": "AUROC",
            "udr": "UDR",
        }
    )

    print("\nPer Dataset Results:")
    print(
        per_dataset_display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nAverage Results Across Datasets:")
    print(
        avg_display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    if avg_df.empty:
        raise RuntimeError(f"No optimizer results were produced: {failed}")

    best_optimizer = avg_df.iloc[0]["optimizer"]
    print(f"\nBest Overall Optimizer: {best_optimizer}")
    print(f"\nSaved per-dataset CSV: {args.csv_path}")
    print(f"Saved average CSV: {args.average_csv_path}")
    print(f"Saved average F1 plot: {args.plot_path}")


if __name__ == "__main__":
    main()
