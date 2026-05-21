"""Probe training and visualisation on cached residual-stream activations."""
from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def load_activations(
    results_dir: str,
    layers: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Load cached per-layer residual-stream tensors.

    Args:
        results_dir: Directory containing acts_layer_{i}.pt files.
        layers: Subset of layer indices to load. Defaults to all available.

    Returns:
        Dict mapping layer index -> float32 array of shape (N, d_model).
    """
    pattern = os.path.join(results_dir, "acts_layer_*.pt")
    out: dict[int, np.ndarray] = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"acts_layer_(\d+)\.pt$", path)
        if m is None:
            continue
        layer = int(m.group(1))
        if layers is not None and layer not in layers:
            continue
        out[layer] = torch.load(path, map_location="cpu").float().numpy()
    return out


def difference_of_means(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean-difference direction between two classes.

    Args:
        acts: (N, D) activations from a single layer.
        labels: (N,) binary labels (0 or 1).

    Returns:
        Unit-norm direction vector (D,), pointing from class 0 to class 1.
    """
    mu1 = acts[labels == 1].mean(axis=0)
    mu0 = acts[labels == 0].mean(axis=0)
    direction = mu1 - mu0
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else direction


def train_logistic_probes(
    acts_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Train a cross-validated logistic probe at each layer.

    C=0.1 (moderate L2) keeps probes from memorising in high dimensions, where
    N (~3.4k) is comparable to d_model (4096).

    Scored with balanced accuracy because SelfAware is ~69/31 answerable/unanswerable:
    plain accuracy gives a misleading 0.69 majority-class baseline, whereas balanced
    accuracy has a clean 0.5 chance level regardless of class skew.

    Args:
        acts_by_layer: Output of load_activations.
        labels: (N,) binary labels.
        cv_folds: Number of CV folds.

    Returns:
        DataFrame with columns [layer, mean_balanced_accuracy, std_balanced_accuracy],
        sorted by layer.
    """
    rows = []
    for layer in sorted(acts_by_layer.keys()):
        X = acts_by_layer[layer]
        clf = LogisticRegression(C=0.1, max_iter=1000)
        scores = cross_val_score(clf, X, labels, cv=cv_folds, scoring="balanced_accuracy")
        rows.append({
            "layer": layer,
            "mean_balanced_accuracy": float(scores.mean()),
            "std_balanced_accuracy": float(scores.std()),
        })
    return pd.DataFrame(rows)


def pca_analysis(
    acts: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
) -> dict:
    """Fit PCA on a single layer's activations.

    Args:
        acts: (N, D) activations.
        labels: (N,) labels passed through for downstream plotting.
        n_components: Number of PCA components.

    Returns:
        Dict with keys: explained_variance_ratio, projected, labels.
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(acts)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "projected": projected,
        "labels": labels,
    }


def plot_probe_accuracy(
    results_df: pd.DataFrame,
    save_path: str = "figures/probe_accuracy_by_layer.png",
) -> None:
    """Headline figure: cross-validated balanced probe accuracy vs layer.

    Args:
        results_df: Output of train_logistic_probes.
        save_path: Output PNG path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results_df["layer"], results_df["mean_balanced_accuracy"], marker="o")
    ax.fill_between(
        results_df["layer"],
        results_df["mean_balanced_accuracy"] - results_df["std_balanced_accuracy"],
        results_df["mean_balanced_accuracy"] + results_df["std_balanced_accuracy"],
        alpha=0.2,
    )
    ax.axhline(0.5, linestyle="--", color="gray", label="chance (balanced)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy (5-fold CV)")
    ax.set_title("Abstention probe accuracy by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
