#!/usr/bin/env python
"""
Production-ready PyTorch inference pipeline for the saved IDS teacher models.

What this script does:
1. Loads a `.pt` checkpoint saved either as a full model or as a `state_dict`
2. Rebuilds the checkpoint architecture automatically when only weights exist
3. Accepts manual samples, CSV files, NumPy arrays, or pandas DataFrames
4. Applies saved preprocessing artifacts when available
5. Falls back to already-preprocessed numeric features when artifacts are absent
6. Runs batch or single-sample inference with `model.eval()` and `torch.no_grad()`
7. Prints readable prediction outputs with confidence, class label, and timing
8. Optionally evaluates accuracy/F1, saves confusion matrices, and tunes a
   softmax-confidence threshold for binary Attack vs Normal decisions

Important note:
The current repository includes the `.pt` teacher checkpoints, but the fitted
feature selector/scaler artifacts are not present by default. Exact training-time
preprocessing is only fully reproducible when files like
`*_feature_selector.pkl`, `*_feature_scaler.pkl`, and `*_le.pkl` exist beside the
checkpoint. If they are missing, this script safely expects numeric features that
already match the checkpoint input dimension.

Example usage:
    python load_model_run/inference_pipeline.py ^
        --model saved_models\\NSL-KDD_teacher.pt ^
        --csv load_model_run\\examples\\nsl_kdd_dummy_input.csv ^
        --label-column label ^
        --print-logits

    python load_model_run/inference_pipeline.py ^
        --model saved_models\\Water_Storage_teacher.pt ^
        --manual-values "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3"
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    aliases: tuple[str, ...]
    default_labels: tuple[str, ...]
    normal_labels: tuple[str, ...]
    label_columns: tuple[str, ...]
    expected_input_dim: Optional[int] = None


DATASET_PROFILES: tuple[DatasetProfile, ...] = (
    DatasetProfile(
        name="NSL-KDD",
        aliases=("nsl-kdd", "nsl_kdd", "nslkdd", "nslkddteacher"),
        default_labels=("dos", "normal", "probe", "r2l"),
        normal_labels=("normal",),
        label_columns=("class", "label", "target"),
        expected_input_dim=20,
    ),
    DatasetProfile(
        name="CICIDS2017",
        aliases=("cicids2017", "cicids_2017", "cicids2017teacher"),
        default_labels=("BENIGN", "Bot", "BruteForce", "DDoS", "Infiltration", "WebAttack"),
        normal_labels=("BENIGN", "Benign", "benign", "normal"),
        label_columns=("Class", "class", "Label", "label", "target"),
        expected_input_dim=20,
    ),
    DatasetProfile(
        name="Gas Pipeline",
        aliases=("gaspipeline", "gas_pipeline", "gaspipelineteacher"),
        default_labels=("0", "1", "2", "3", "4", "5", "7"),
        normal_labels=("0", "normal", "Normal"),
        label_columns=("result", "label", "target"),
        expected_input_dim=20,
    ),
    DatasetProfile(
        name="Water Storage",
        aliases=("waterstorage", "water_storage", "waterstorageteacher"),
        default_labels=("0", "2", "3", "4", "5", "6", "7"),
        normal_labels=("0", "normal", "Normal"),
        label_columns=("result", "label", "target"),
        expected_input_dim=23,
    ),
)


@dataclass
class ArchitectureSpec:
    input_dim: int
    latent_dim: int
    output_dim: int
    num_classes: int
    encoder_hidden_dims: list[int]
    decoder_hidden_dims: list[int]
    classifier_hidden_dims: list[int]


@dataclass
class ArtifactBundle:
    feature_selector: Any = None
    feature_scaler: Any = None
    label_encoder: Any = None
    label_names: Optional[list[str]] = None
    source_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class ModelBundle:
    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    architecture: ArchitectureSpec
    dataset_profile: Optional[DatasetProfile]
    class_names: list[str]
    normal_labels: set[str]
    artifacts: ArtifactBundle
    source_type: str


class DynamicEncoder(nn.Module):
    """Checkpoint-driven encoder that matches saved state_dict layer names."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], latent_dim: int) -> None:
        super().__init__()
        prev_dim = input_dim
        self.hidden_layer_names: list[str] = []
        for idx, hidden_dim in enumerate(hidden_dims, start=1):
            layer_name = f"fc{idx}"
            setattr(self, layer_name, nn.Linear(prev_dim, hidden_dim))
            self.hidden_layer_names.append(layer_name)
            prev_dim = hidden_dim
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer_name in self.hidden_layer_names:
            x = self.activation(getattr(self, layer_name)(x))
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), min=-10.0, max=10.0)
        return mu, logvar


class DynamicDecoder(nn.Module):
    """Checkpoint-driven decoder that matches saved state_dict layer names."""

    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        prev_dim = latent_dim
        self.hidden_layer_names: list[str] = []
        for idx, hidden_dim in enumerate(hidden_dims, start=1):
            layer_name = f"fc{idx}"
            setattr(self, layer_name, nn.Linear(prev_dim, hidden_dim))
            self.hidden_layer_names.append(layer_name)
            prev_dim = hidden_dim
        self.output_layer_name = f"fc{len(hidden_dims) + 1}"
        setattr(self, self.output_layer_name, nn.Linear(prev_dim, output_dim))
        self.activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer_name in self.hidden_layer_names:
            z = self.activation(getattr(self, layer_name)(z))
        return getattr(self, self.output_layer_name)(z)


class DynamicTeacherClassifier(nn.Module):
    """Checkpoint-driven classifier preserving `classifier.net.*` key layout."""

    def __init__(self, latent_dim: int, hidden_dims: Sequence[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DynamicVAEWithTeacher(nn.Module):
    """
    Generic VAE+Teacher model rebuilt entirely from checkpoint tensor shapes.
    """

    def __init__(self, architecture: ArchitectureSpec) -> None:
        super().__init__()
        self.encoder = DynamicEncoder(
            input_dim=architecture.input_dim,
            hidden_dims=architecture.encoder_hidden_dims,
            latent_dim=architecture.latent_dim,
        )
        self.decoder = DynamicDecoder(
            latent_dim=architecture.latent_dim,
            hidden_dims=architecture.decoder_hidden_dims,
            output_dim=architecture.output_dim,
        )
        self.classifier = DynamicTeacherClassifier(
            latent_dim=architecture.latent_dim,
            hidden_dims=architecture.classifier_hidden_dims,
            num_classes=architecture.num_classes,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        logits = self.classifier(mu)
        return recon, mu, logvar, logits


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_compat(path: Path, map_location: torch.device, weights_only: Optional[bool] = None) -> Any:
    load_kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters and weights_only is not None:
        load_kwargs["weights_only"] = weights_only
    return torch.load(path, **load_kwargs)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _is_state_dict(candidate: Any) -> bool:
    if not isinstance(candidate, dict) or not candidate:
        return False
    tensor_like = [torch.is_tensor(value) for value in candidate.values()]
    return bool(tensor_like) and all(tensor_like)


def _extract_state_dict(checkpoint_obj: Any) -> Optional[dict[str, torch.Tensor]]:
    if isinstance(checkpoint_obj, nn.Module):
        return checkpoint_obj.state_dict()

    if _is_state_dict(checkpoint_obj):
        return checkpoint_obj

    if not isinstance(checkpoint_obj, dict):
        return None

    preferred_keys = (
        "state_dict",
        "model_state_dict",
        "teacher_state_dict",
        "net",
        "model",
        "teacher",
    )
    for key in preferred_keys:
        candidate = checkpoint_obj.get(key)
        if _is_state_dict(candidate):
            return candidate
        if isinstance(candidate, nn.Module):
            return candidate.state_dict()

    for value in checkpoint_obj.values():
        if _is_state_dict(value):
            return value

    return None


def infer_architecture_from_state_dict(state_dict: dict[str, torch.Tensor]) -> ArchitectureSpec:
    state_dict = _strip_module_prefix(state_dict)

    encoder_hidden: list[tuple[int, torch.Tensor]] = []
    decoder_layers: list[tuple[int, torch.Tensor]] = []
    classifier_layers: list[tuple[int, torch.Tensor]] = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value) or value.ndim != 2:
            continue

        encoder_match = re.fullmatch(r"encoder\.fc(\d+)\.weight", key)
        decoder_match = re.fullmatch(r"decoder\.fc(\d+)\.weight", key)
        classifier_match = re.fullmatch(r"classifier\.net\.(\d+)\.weight", key)

        if encoder_match:
            encoder_hidden.append((int(encoder_match.group(1)), value))
        elif decoder_match:
            decoder_layers.append((int(decoder_match.group(1)), value))
        elif classifier_match:
            classifier_layers.append((int(classifier_match.group(1)), value))

    if "encoder.fc_mu.weight" not in state_dict or "encoder.fc_logvar.weight" not in state_dict:
        raise ValueError("Checkpoint does not contain the expected VAE latent heads.")

    if not encoder_hidden or not decoder_layers or not classifier_layers:
        raise ValueError("Checkpoint is missing encoder/decoder/classifier weights.")

    encoder_hidden.sort(key=lambda item: item[0])
    decoder_layers.sort(key=lambda item: item[0])
    classifier_layers.sort(key=lambda item: item[0])

    input_dim = int(encoder_hidden[0][1].shape[1])
    encoder_hidden_dims = [int(weight.shape[0]) for _, weight in encoder_hidden]
    latent_dim = int(state_dict["encoder.fc_mu.weight"].shape[0])

    decoder_hidden_dims = [int(weight.shape[0]) for _, weight in decoder_layers[:-1]]
    output_dim = int(decoder_layers[-1][1].shape[0])

    classifier_hidden_dims = [int(weight.shape[0]) for _, weight in classifier_layers[:-1]]
    num_classes = int(classifier_layers[-1][1].shape[0])

    return ArchitectureSpec(
        input_dim=input_dim,
        latent_dim=latent_dim,
        output_dim=output_dim,
        num_classes=num_classes,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        classifier_hidden_dims=classifier_hidden_dims,
    )


def _match_dataset_profile(dataset_name: Optional[str], checkpoint_path: Path) -> Optional[DatasetProfile]:
    candidates = []
    if dataset_name:
        candidates.append(_normalize_name(dataset_name))
    candidates.append(_normalize_name(checkpoint_path.stem))
    candidates.append(_normalize_name(checkpoint_path.name))

    for candidate in candidates:
        for profile in DATASET_PROFILES:
            profile_tokens = {_normalize_name(profile.name), *(_normalize_name(alias) for alias in profile.aliases)}
            if candidate in profile_tokens or any(alias in candidate for alias in profile_tokens):
                return profile
    return None


def _discover_artifact_paths(checkpoint_path: Path, artifacts_dir: Path) -> dict[str, Path]:
    stem = checkpoint_path.stem
    dataset_key = stem.replace("_teacher", "").replace("-teacher", "")
    patterns = {
        "feature_selector": [
            f"{stem}_feature_selector.pkl",
            f"{dataset_key}_feature_selector.pkl",
            f"{dataset_key}_selector.pkl",
        ],
        "feature_scaler": [
            f"{stem}_feature_scaler.pkl",
            f"{dataset_key}_feature_scaler.pkl",
            f"{dataset_key}_scaler.pkl",
        ],
        "label_encoder": [
            f"{stem}_le.pkl",
            f"{dataset_key}_le.pkl",
            f"{stem}_label_encoder.pkl",
            f"{dataset_key}_label_encoder.pkl",
        ],
        "label_names_json": [
            f"{stem}_labels.json",
            f"{dataset_key}_labels.json",
        ],
    }

    resolved: dict[str, Path] = {}
    for key, names in patterns.items():
        for name in names:
            candidate = artifacts_dir / name
            if candidate.exists():
                resolved[key] = candidate
                break
    return resolved


def _load_pickle_like(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def _resolve_label_names(
    architecture: ArchitectureSpec,
    dataset_profile: Optional[DatasetProfile],
    artifacts: ArtifactBundle,
    class_labels_override: Optional[str] = None,
) -> list[str]:
    if class_labels_override:
        labels = [item.strip() for item in class_labels_override.split(",") if item.strip()]
        if len(labels) != architecture.num_classes:
            raise ValueError(
                f"--class-labels provided {len(labels)} labels, but checkpoint expects "
                f"{architecture.num_classes} classes."
            )
        return labels

    if artifacts.label_names:
        if len(artifacts.label_names) != architecture.num_classes:
            raise ValueError(
                "Loaded label metadata does not match the checkpoint class count "
                f"({len(artifacts.label_names)} != {architecture.num_classes})."
            )
        return [str(label) for label in artifacts.label_names]

    if artifacts.label_encoder is not None and hasattr(artifacts.label_encoder, "classes_"):
        classes = [str(label) for label in artifacts.label_encoder.classes_]
        if len(classes) == architecture.num_classes:
            return classes

    if dataset_profile and len(dataset_profile.default_labels) == architecture.num_classes:
        warnings.warn(
            "No saved label encoder was found beside the checkpoint. "
            f"Using built-in default labels for {dataset_profile.name}.",
            stacklevel=2,
        )
        return list(dataset_profile.default_labels)

    return [f"class_{idx}" for idx in range(architecture.num_classes)]


def load_model(
    model_path: str | Path,
    device: str = "auto",
    artifacts_dir: Optional[str | Path] = None,
    dataset_name: Optional[str] = None,
    class_labels_override: Optional[str] = None,
    normal_labels_override: Optional[str] = None,
) -> ModelBundle:
    """
    Load a saved teacher checkpoint and rebuild the architecture when needed.
    """

    checkpoint_path = Path(model_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch_device = resolve_device(device)
    raw_obj: Any
    source_type = "unknown"

    try:
        raw_obj = _torch_load_compat(checkpoint_path, map_location=torch_device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint '{checkpoint_path.name}': {exc}") from exc

    if isinstance(raw_obj, nn.Module):
        model = raw_obj.to(torch_device)
        model.eval()
        state_dict = _strip_module_prefix(model.state_dict())
        architecture = infer_architecture_from_state_dict(state_dict)
        source_type = "full_model"
    else:
        state_dict = _extract_state_dict(raw_obj)
        if state_dict is None:
            raise TypeError(
                f"Unsupported checkpoint format in {checkpoint_path.name}. "
                "Expected a full `nn.Module`, a `state_dict`, or a checkpoint dict "
                "containing one of them."
            )
        state_dict = _strip_module_prefix(state_dict)
        architecture = infer_architecture_from_state_dict(state_dict)
        model = DynamicVAEWithTeacher(architecture)
        model.load_state_dict(state_dict, strict=True)
        model.to(torch_device)
        model.eval()
        source_type = "state_dict"

    dataset_profile = _match_dataset_profile(dataset_name, checkpoint_path)
    if (
        dataset_profile is not None
        and dataset_profile.expected_input_dim is not None
        and dataset_profile.expected_input_dim != architecture.input_dim
    ):
        warnings.warn(
            f"Dataset profile '{dataset_profile.name}' usually expects {dataset_profile.expected_input_dim} "
            f"features, but checkpoint '{checkpoint_path.name}' expects {architecture.input_dim}. "
            "The checkpoint shape will be trusted.",
            stacklevel=2,
        )
    artifact_dir_path = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir
        else checkpoint_path.parent.resolve()
    )
    artifact_paths = _discover_artifact_paths(checkpoint_path, artifact_dir_path)

    artifacts = ArtifactBundle(source_paths=artifact_paths.copy())
    if "feature_selector" in artifact_paths:
        artifacts.feature_selector = _load_pickle_like(artifact_paths["feature_selector"])
    if "feature_scaler" in artifact_paths:
        artifacts.feature_scaler = _load_pickle_like(artifact_paths["feature_scaler"])
    if "label_encoder" in artifact_paths:
        artifacts.label_encoder = _load_pickle_like(artifact_paths["label_encoder"])
    if "label_names_json" in artifact_paths:
        with artifact_paths["label_names_json"].open("r", encoding="utf-8") as handle:
            loaded_labels = json.load(handle)
        if isinstance(loaded_labels, dict):
            loaded_labels = loaded_labels.get("labels", loaded_labels.get("classes"))
        if not isinstance(loaded_labels, list):
            raise ValueError(f"Label JSON must contain a list of class names: {artifact_paths['label_names_json']}")
        artifacts.label_names = [str(label) for label in loaded_labels]

    class_names = _resolve_label_names(
        architecture=architecture,
        dataset_profile=dataset_profile,
        artifacts=artifacts,
        class_labels_override=class_labels_override,
    )

    if normal_labels_override:
        normal_labels = {item.strip() for item in normal_labels_override.split(",") if item.strip()}
    elif dataset_profile:
        normal_labels = set(dataset_profile.normal_labels)
    else:
        normal_labels = {"normal", "Normal", "BENIGN", "benign", "Benign", "0"}

    return ModelBundle(
        model=model,
        device=torch_device,
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        dataset_profile=dataset_profile,
        class_names=class_names,
        normal_labels=normal_labels,
        artifacts=artifacts,
        source_type=source_type,
    )


def _read_input_object(input_data: Any) -> tuple[np.ndarray | pd.DataFrame, str]:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy(), "dataframe"

    if isinstance(input_data, np.ndarray):
        return np.asarray(input_data), "ndarray"

    if isinstance(input_data, (list, tuple)):
        return np.asarray(input_data, dtype=np.float32), "manual"

    if isinstance(input_data, (str, Path)):
        path = Path(input_data).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path), "csv"
        if suffix == ".npy":
            return np.load(path), "npy"
        if suffix == ".npz":
            archive = np.load(path)
            first_key = archive.files[0]
            return archive[first_key], f"npz:{first_key}"
        raise ValueError(f"Unsupported input file format: {path.suffix}. Use CSV, NPY, or NPZ.")

    raise TypeError(
        "Unsupported input type. Use a pandas DataFrame, NumPy array, list/tuple, "
        "CSV path, or NPY/NPZ path."
    )


def _coerce_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0)

    non_numeric_columns = [
        column for column in cleaned.columns if not pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns were found: "
            f"{non_numeric_columns}. Exact categorical encoders were not saved in this repo, "
            "so please provide numeric features that were already encoded the same way as training."
        )

    return cleaned.apply(pd.to_numeric, errors="coerce").fillna(0)


def _apply_selector_if_needed(
    X: np.ndarray,
    selector: Any,
    expected_input_dim: int,
) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if selector is None:
        return X, notes

    selector_input_dim = getattr(selector, "original_feature_count", None)
    selected_dim = len(getattr(selector, "selected_indices", [])) or getattr(selector, "k_features", None)

    if selector_input_dim is not None and X.shape[1] == selector_input_dim:
        X = selector.transform(X)
        notes.append(f"Applied saved feature selector: {selector_input_dim} -> {X.shape[1]} features")
        return X, notes

    if X.shape[1] == expected_input_dim:
        notes.append("Skipped feature selector because input already matches checkpoint feature dimension")
        return X, notes

    if selected_dim is not None and X.shape[1] == selected_dim:
        notes.append("Skipped feature selector because input already matches selected feature count")
        return X, notes

    raise ValueError(
        "Feature selector exists, but the input dimension does not match either the selector's "
        f"expected raw feature count ({selector_input_dim}) or the checkpoint input dimension "
        f"({expected_input_dim}). Received {X.shape[1]} features."
    )


def _apply_scaler_if_needed(
    X: np.ndarray,
    scaler: Any,
    fit_fallback_scaler: bool,
) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    if scaler is not None:
        if hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) != X.shape[1]:
            raise ValueError(
                f"Saved scaler expects {int(scaler.n_features_in_)} features, but received {X.shape[1]}."
            )
        X = scaler.transform(X).astype(np.float32, copy=False)
        notes.append("Applied saved feature scaler")
        return X, notes

    if fit_fallback_scaler:
        fallback_scaler = StandardScaler()
        X = fallback_scaler.fit_transform(X).astype(np.float32, copy=False)
        notes.append("Applied fallback StandardScaler fit on the provided input data")
        warnings.warn(
            "No saved scaler artifact was found. A fallback StandardScaler was fit on the input data. "
            "This is convenient for demos, but it is not the exact training-time scaler.",
            stacklevel=2,
        )
        return X, notes

    return X.astype(np.float32, copy=False), notes


def _resolve_label_column(
    df: pd.DataFrame,
    explicit_label_column: Optional[str],
    dataset_profile: Optional[DatasetProfile],
) -> Optional[str]:
    if explicit_label_column:
        if explicit_label_column not in df.columns:
            raise ValueError(f"Requested label column '{explicit_label_column}' is not present in the input file.")
        return explicit_label_column

    candidate_columns = dataset_profile.label_columns if dataset_profile else ("label", "Label", "class", "Class")
    for candidate in candidate_columns:
        if candidate in df.columns:
            return candidate
    return None


def preprocess_input(
    input_data: Any,
    bundle: ModelBundle,
    label_column: Optional[str] = None,
    fit_fallback_scaler: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray], list[str], str]:
    """
    Convert manual/CSV/NumPy/DataFrame input into a model-ready float32 matrix.
    """

    raw_input, source_kind = _read_input_object(input_data)
    notes: list[str] = []
    y_true: Optional[np.ndarray] = None

    if isinstance(raw_input, pd.DataFrame):
        resolved_label_column = _resolve_label_column(raw_input, label_column, bundle.dataset_profile)
        features_df = raw_input.copy()
        if resolved_label_column:
            y_true = features_df[resolved_label_column].astype(str).to_numpy()
            features_df = features_df.drop(columns=[resolved_label_column])
            notes.append(f"Separated label column: {resolved_label_column}")

        features_df = _coerce_numeric_dataframe(features_df)
        X = features_df.to_numpy(dtype=np.float32)
    else:
        X = np.asarray(raw_input, dtype=np.float32)

    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
        notes.append("Converted single sample into batch shape [1, num_features]")
    elif X.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D feature array, but received shape {X.shape}.")

    if bundle.artifacts.feature_selector is not None:
        X, selector_notes = _apply_selector_if_needed(
            X,
            selector=bundle.artifacts.feature_selector,
            expected_input_dim=bundle.architecture.input_dim,
        )
        notes.extend(selector_notes)

    if bundle.artifacts.feature_scaler is not None or fit_fallback_scaler:
        X, scaler_notes = _apply_scaler_if_needed(
            X,
            scaler=bundle.artifacts.feature_scaler,
            fit_fallback_scaler=fit_fallback_scaler,
        )
        notes.extend(scaler_notes)
    else:
        X = X.astype(np.float32, copy=False)
        if X.shape[1] == bundle.architecture.input_dim:
            notes.append("No scaler artifact found; treating input as already preprocessed numeric features")

    if X.shape[1] != bundle.architecture.input_dim:
        raise ValueError(
            f"Checkpoint expects {bundle.architecture.input_dim} input features, but received {X.shape[1]}."
        )

    return X.astype(np.float32, copy=False), y_true, notes, source_kind


def decode_label(class_id: int, class_names: Sequence[str]) -> str:
    if class_id < 0 or class_id >= len(class_names):
        raise IndexError(f"Class index {class_id} is outside the available label range [0, {len(class_names) - 1}].")
    return str(class_names[class_id])


def _binary_attack_normal(label: str, normal_labels: set[str]) -> str:
    return "Normal" if str(label) in normal_labels else "Attack"


def _forward_to_logits(model: nn.Module, features: torch.Tensor) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    outputs = model(features)

    if isinstance(outputs, tuple):
        if len(outputs) >= 4:
            recon, _, _, logits = outputs[:4]
            return recon, logits
        if len(outputs) == 2:
            first, second = outputs
            if first.ndim == features.ndim and second.ndim == 2:
                return first, second
            if second.ndim == features.ndim and first.ndim == 2:
                return second, first
            return None, second if second.ndim == 2 else first
        if len(outputs) == 1:
            output = outputs[0]
            return None, output

    if not torch.is_tensor(outputs):
        raise TypeError("Model forward pass did not return a tensor or a tuple of tensors.")
    return None, outputs


def predict_batch(
    bundle: ModelBundle,
    input_data: Any,
    batch_size: int = 256,
    label_column: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    fit_fallback_scaler: bool = False,
) -> dict[str, Any]:
    """
    Run batched inference and return predictions, probabilities, and metadata.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    X, y_true, preprocessing_notes, source_kind = preprocess_input(
        input_data=input_data,
        bundle=bundle,
        label_column=label_column,
        fit_fallback_scaler=fit_fallback_scaler,
    )

    tensor_x = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)

    results: list[dict[str, Any]] = []
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    recon_errors_all: list[np.ndarray] = []
    batch_times: list[float] = []

    bundle.model.eval()
    total_start = time.perf_counter()

    with torch.no_grad():
        sample_offset = 0
        for (batch_features,) in loader:
            batch_features = batch_features.to(bundle.device, dtype=torch.float32, non_blocking=bundle.device.type == "cuda")
            batch_start = time.perf_counter()
            recon, logits = _forward_to_logits(bundle.model, batch_features)
            batch_elapsed = time.perf_counter() - batch_start
            batch_times.append(batch_elapsed)

            probabilities = torch.softmax(logits, dim=1)
            confidences, class_ids = torch.max(probabilities, dim=1)

            batch_logits = logits.detach().cpu().numpy()
            batch_probs = probabilities.detach().cpu().numpy()
            batch_confidences = confidences.detach().cpu().numpy()
            batch_class_ids = class_ids.detach().cpu().numpy()

            if recon is not None and recon.shape == batch_features.shape:
                batch_recon_errors = (
                    F.mse_loss(recon, batch_features, reduction="none").mean(dim=1).detach().cpu().numpy()
                )
            else:
                batch_recon_errors = np.full(shape=(batch_features.shape[0],), fill_value=np.nan, dtype=np.float32)

            logits_all.append(batch_logits)
            probs_all.append(batch_probs)
            recon_errors_all.append(batch_recon_errors)

            per_sample_time = batch_elapsed / max(1, len(batch_class_ids))
            for local_index, (class_id, confidence, raw_logits, prob_vector, recon_error) in enumerate(
                zip(batch_class_ids, batch_confidences, batch_logits, batch_probs, batch_recon_errors)
            ):
                predicted_label = decode_label(int(class_id), bundle.class_names)
                low_confidence = confidence_threshold is not None and float(confidence) < float(confidence_threshold)

                results.append(
                    {
                        "sample_index": sample_offset + local_index + 1,
                        "predicted_class": predicted_label,
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "attack_normal_label": _binary_attack_normal(predicted_label, bundle.normal_labels),
                        "raw_logits": raw_logits.tolist(),
                        "probabilities": prob_vector.tolist(),
                        "reconstruction_error": float(recon_error) if not np.isnan(recon_error) else None,
                        "inference_time_sec": float(per_sample_time),
                        "low_confidence": low_confidence,
                    }
                )
            sample_offset += len(batch_class_ids)

    total_elapsed = time.perf_counter() - total_start

    return {
        "results": results,
        "X": X,
        "y_true": y_true,
        "logits": np.vstack(logits_all) if logits_all else np.empty((0, bundle.architecture.num_classes), dtype=np.float32),
        "probabilities": np.vstack(probs_all) if probs_all else np.empty((0, bundle.architecture.num_classes), dtype=np.float32),
        "reconstruction_errors": np.concatenate(recon_errors_all) if recon_errors_all else np.empty((0,), dtype=np.float32),
        "source_kind": source_kind,
        "preprocessing_notes": preprocessing_notes,
        "total_inference_time_sec": float(total_elapsed),
        "average_batch_time_sec": float(np.mean(batch_times)) if batch_times else 0.0,
    }


def predict(
    bundle: ModelBundle,
    input_data: Any,
    label_column: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    fit_fallback_scaler: bool = False,
) -> dict[str, Any]:
    """
    Single-sample prediction helper.
    """

    payload = predict_batch(
        bundle=bundle,
        input_data=input_data,
        batch_size=1,
        label_column=label_column,
        confidence_threshold=confidence_threshold,
        fit_fallback_scaler=fit_fallback_scaler,
    )
    return payload["results"][0]


def save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 0.9)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def evaluate_predictions(
    y_true: Sequence[str],
    batch_payload: dict[str, Any],
    bundle: ModelBundle,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Compute multiclass and binary Attack/Normal metrics plus confusion matrices.
    """

    y_true_arr = np.asarray([str(item) for item in y_true])
    pred_labels = np.asarray([row["predicted_class"] for row in batch_payload["results"]], dtype=object)

    metrics = {
        "multiclass_accuracy": float(accuracy_score(y_true_arr, pred_labels)),
        "multiclass_precision_weighted": float(
            precision_score(y_true_arr, pred_labels, average="weighted", zero_division=0)
        ),
        "multiclass_recall_weighted": float(
            recall_score(y_true_arr, pred_labels, average="weighted", zero_division=0)
        ),
        "multiclass_f1_weighted": float(f1_score(y_true_arr, pred_labels, average="weighted", zero_division=0)),
        "multiclass_f1_macro": float(f1_score(y_true_arr, pred_labels, average="macro", zero_division=0)),
    }

    y_true_binary = np.asarray([_binary_attack_normal(label, bundle.normal_labels) for label in y_true_arr], dtype=object)
    y_pred_binary = np.asarray(
        [row["attack_normal_label"] for row in batch_payload["results"]],
        dtype=object,
    )

    metrics.update(
        {
            "binary_accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
            "binary_precision": float(
                precision_score(y_true_binary, y_pred_binary, pos_label="Attack", zero_division=0)
            ),
            "binary_recall": float(
                recall_score(y_true_binary, y_pred_binary, pos_label="Attack", zero_division=0)
            ),
            "binary_f1": float(
                f1_score(y_true_binary, y_pred_binary, pos_label="Attack", zero_division=0)
            ),
        }
    )

    saved_paths: dict[str, str] = {}
    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        multiclass_labels = sorted(set(y_true_arr.tolist()) | set(pred_labels.tolist()))
        multiclass_path = save_confusion_matrix(
            y_true=y_true_arr,
            y_pred=pred_labels,
            labels=multiclass_labels,
            output_path=output_dir / "confusion_matrix_multiclass.png",
            title=f"{bundle.checkpoint_path.stem} - Multiclass Confusion Matrix",
        )
        binary_path = save_confusion_matrix(
            y_true=y_true_binary,
            y_pred=y_pred_binary,
            labels=["Normal", "Attack"],
            output_path=output_dir / "confusion_matrix_attack_normal.png",
            title=f"{bundle.checkpoint_path.stem} - Attack vs Normal",
        )
        saved_paths["multiclass_confusion_matrix"] = str(multiclass_path)
        saved_paths["binary_confusion_matrix"] = str(binary_path)

    return {"metrics": metrics, "saved_paths": saved_paths}


def tune_probability_threshold(
    y_true: Sequence[str],
    batch_payload: dict[str, Any],
    bundle: ModelBundle,
    thresholds: Optional[Iterable[float]] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Tune a softmax confidence threshold for binary Attack/Normal classification.

    Decision rule:
    - If the predicted class is a normal class, require confidence >= threshold
      to keep the sample as Normal.
    - Otherwise label it as Attack.
    """

    thresholds = list(thresholds) if thresholds is not None else np.round(np.linspace(0.50, 0.99, 50), 2).tolist()
    y_true_arr = np.asarray([str(item) for item in y_true], dtype=object)
    y_true_binary = np.asarray([_binary_attack_normal(label, bundle.normal_labels) for label in y_true_arr], dtype=object)

    predicted_labels = [row["predicted_class"] for row in batch_payload["results"]]
    confidences = [float(row["confidence"]) for row in batch_payload["results"]]

    sweep_rows: list[dict[str, float]] = []
    best_row: Optional[dict[str, float]] = None

    for threshold in thresholds:
        threshold = float(threshold)
        binary_predictions = []
        for predicted_label, confidence in zip(predicted_labels, confidences):
            if predicted_label in bundle.normal_labels and confidence >= threshold:
                binary_predictions.append("Normal")
            else:
                binary_predictions.append("Attack")

        acc = float(accuracy_score(y_true_binary, binary_predictions))
        prec = float(precision_score(y_true_binary, binary_predictions, pos_label="Attack", zero_division=0))
        rec = float(recall_score(y_true_binary, binary_predictions, pos_label="Attack", zero_division=0))
        f1 = float(f1_score(y_true_binary, binary_predictions, pos_label="Attack", zero_division=0))

        row = {
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        sweep_rows.append(row)

        if best_row is None or (row["f1"], row["accuracy"], -abs(row["threshold"] - 0.75)) > (
            best_row["f1"],
            best_row["accuracy"],
            -abs(best_row["threshold"] - 0.75),
        ):
            best_row = row

    if best_row is None:
        raise RuntimeError("Threshold tuning failed because no thresholds were evaluated.")

    saved_path = None
    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = output_dir / "threshold_tuning.csv"
        pd.DataFrame(sweep_rows).to_csv(saved_path, index=False)

    return {"best": best_row, "sweep": sweep_rows, "saved_path": str(saved_path) if saved_path else None}


def write_results_csv(batch_payload: dict[str, Any], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(batch_payload["results"]).to_csv(output_csv, index=False)
    return output_csv


def _print_bundle_summary(bundle: ModelBundle) -> None:
    print("=" * 72)
    print("IDS Inference Pipeline")
    print("=" * 72)
    print(f"Checkpoint         : {bundle.checkpoint_path}")
    print(f"Checkpoint Type    : {bundle.source_type}")
    print(f"Dataset            : {bundle.dataset_profile.name if bundle.dataset_profile else 'Auto/Unknown'}")
    print(f"Device             : {bundle.device}")
    if bundle.device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print(f"Input Features     : {bundle.architecture.input_dim}")
    print(f"Latent Dim         : {bundle.architecture.latent_dim}")
    print(f"Number of Classes  : {bundle.architecture.num_classes}")
    print(f"Class Labels       : {bundle.class_names}")

    if bundle.artifacts.source_paths:
        print("Artifacts          :")
        for name, path in bundle.artifacts.source_paths.items():
            print(f"  - {name}: {path}")
    else:
        print("Artifacts          : none found beside checkpoint")
        print("                     expecting already-preprocessed numeric features")
    print("-" * 72)


def _print_preprocessing_notes(notes: Sequence[str]) -> None:
    if not notes:
        return
    print("Preprocessing:")
    for note in notes:
        print(f"  - {note}")
    print("-" * 72)


def _print_predictions(
    batch_payload: dict[str, Any],
    print_logits: bool = False,
    print_probabilities: bool = False,
    confidence_threshold: Optional[float] = None,
) -> None:
    for row in batch_payload["results"]:
        print("-" * 56)
        print(f"Sample {row['sample_index']}:")
        print(f"Predicted Class : {row['predicted_class']}")
        print(f"Confidence      : {row['confidence'] * 100:.2f}%")
        print(f"Class ID        : {row['class_id']}")
        print(f"Attack/Normal   : {row['attack_normal_label']}")
        if row["reconstruction_error"] is not None:
            print(f"Recon Error     : {row['reconstruction_error']:.6f}")
        print(f"Inference Time  : {row['inference_time_sec']:.6f} sec")
        if confidence_threshold is not None:
            status = "Below threshold" if row["low_confidence"] else "Passed threshold"
            print(f"Conf Threshold  : {status} ({confidence_threshold:.2f})")
        if print_logits:
            print(f"Raw Logits      : {np.array2string(np.asarray(row['raw_logits']), precision=5)}")
        if print_probabilities:
            print(f"Probabilities   : {np.array2string(np.asarray(row['probabilities']), precision=5)}")
    print("-" * 56)
    print(f"Total Samples   : {len(batch_payload['results'])}")
    print(f"Total Time      : {batch_payload['total_inference_time_sec']:.6f} sec")
    print(f"Avg Batch Time  : {batch_payload['average_batch_time_sec']:.6f} sec")
    print("-" * 56)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a saved IDS teacher checkpoint and run inference on manual, CSV, NumPy, or DataFrame inputs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a saved .pt checkpoint such as saved_models/NSL-KDD_teacher.pt",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name override (e.g. NSL-KDD, CICIDS2017, Gas Pipeline, Water Storage).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory that contains saved preprocessing artifacts (*.pkl / *.json). Defaults to the model directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' selects GPU when available.",
    )
    parser.add_argument(
        "--manual-values",
        default=None,
        help="Comma-separated manual single-sample feature values.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to a CSV input file.",
    )
    parser.add_argument(
        "--npy",
        default=None,
        help="Path to a .npy or .npz NumPy input file.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional ground-truth label column for evaluation when using CSV/DataFrame input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference. Default: 256",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Optional softmax threshold used to flag low-confidence predictions.",
    )
    parser.add_argument(
        "--fit-fallback-scaler",
        action="store_true",
        help="Fit a StandardScaler on the provided input if no saved scaler artifact is available.",
    )
    parser.add_argument(
        "--class-labels",
        default=None,
        help="Optional comma-separated override for class names in checkpoint order.",
    )
    parser.add_argument(
        "--normal-labels",
        default=None,
        help="Optional comma-separated override for labels that should be treated as Normal.",
    )
    parser.add_argument(
        "--print-logits",
        action="store_true",
        help="Print raw logits for each prediction.",
    )
    parser.add_argument(
        "--print-probabilities",
        action="store_true",
        help="Print the full softmax probability vector for each prediction.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute accuracy/F1 metrics when ground-truth labels are available.",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Sweep softmax thresholds and report the best binary Attack/Normal threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for confusion matrices and threshold tuning CSV output.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Optional path to save the prediction table as CSV.",
    )
    return parser


def _parse_manual_values(manual_values: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in manual_values.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--manual-values must be a comma-separated list of numbers.") from exc
    if not values:
        raise ValueError("--manual-values was provided, but no numeric values were parsed.")
    return np.asarray(values, dtype=np.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    provided_inputs = [args.manual_values is not None, args.csv is not None, args.npy is not None]
    if sum(provided_inputs) != 1:
        parser.error("Provide exactly one input source: --manual-values, --csv, or --npy.")

    input_data: Any
    if args.manual_values is not None:
        input_data = _parse_manual_values(args.manual_values)
    elif args.csv is not None:
        input_data = args.csv
    else:
        input_data = args.npy

    try:
        bundle = load_model(
            model_path=args.model,
            device=args.device,
            artifacts_dir=args.artifacts_dir,
            dataset_name=args.dataset,
            class_labels_override=args.class_labels,
            normal_labels_override=args.normal_labels,
        )
        _print_bundle_summary(bundle)

        batch_payload = predict_batch(
            bundle=bundle,
            input_data=input_data,
            batch_size=1 if args.manual_values is not None else args.batch_size,
            label_column=args.label_column,
            confidence_threshold=args.confidence_threshold,
            fit_fallback_scaler=args.fit_fallback_scaler,
        )

        _print_preprocessing_notes(batch_payload["preprocessing_notes"])
        _print_predictions(
            batch_payload=batch_payload,
            print_logits=args.print_logits,
            print_probabilities=args.print_probabilities,
            confidence_threshold=args.confidence_threshold,
        )

        if args.results_csv:
            results_path = write_results_csv(batch_payload, args.results_csv)
            print(f"Saved prediction table: {results_path}")

        if args.evaluate:
            if batch_payload["y_true"] is None:
                raise ValueError(
                    "--evaluate was requested, but no ground-truth labels were found. "
                    "Add a label column to the CSV and pass --label-column if needed."
                )
            evaluation = evaluate_predictions(
                y_true=batch_payload["y_true"],
                batch_payload=batch_payload,
                bundle=bundle,
                output_dir=args.output_dir,
            )
            print("Evaluation Metrics:")
            for key, value in evaluation["metrics"].items():
                print(f"  - {key}: {value:.6f}")
            for name, path in evaluation["saved_paths"].items():
                print(f"Saved {name}: {path}")

        if args.tune_threshold:
            if batch_payload["y_true"] is None:
                raise ValueError(
                    "--tune-threshold was requested, but no ground-truth labels were found. "
                    "Add a label column to the CSV and pass --label-column if needed."
                )
            tuning = tune_probability_threshold(
                y_true=batch_payload["y_true"],
                batch_payload=batch_payload,
                bundle=bundle,
                output_dir=args.output_dir,
            )
            best = tuning["best"]
            print("Best Threshold:")
            print(f"  - threshold: {best['threshold']:.2f}")
            print(f"  - accuracy : {best['accuracy']:.6f}")
            print(f"  - precision: {best['precision']:.6f}")
            print(f"  - recall   : {best['recall']:.6f}")
            print(f"  - f1       : {best['f1']:.6f}")
            if tuning["saved_path"]:
                print(f"Saved threshold sweep: {tuning['saved_path']}")

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
