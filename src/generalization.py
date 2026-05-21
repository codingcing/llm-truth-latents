"""Cross-topic generalisation of the abstention probe."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _label_column(labels_df: pd.DataFrame) -> str:
    """Pick the first non-topic column as the label column."""
    for col in labels_df.columns:
        if col != "topic":
            return col
    raise ValueError("labels_df must contain a non-'topic' column to use as the label.")


def cross_topic_eval(
    acts_by_layer: dict[int, np.ndarray],
    labels_df: pd.DataFrame,
    held_out_topics: list[str],
    peak_layer: int,
) -> dict:
    """Train a probe on non-held-out topics; evaluate on the held-out set.

    Args:
        acts_by_layer: Dict from probes.load_activations.
        labels_df: DataFrame with columns [topic, <label_col>], aligned row-wise to acts.
            The first non-topic column is treated as the binary label.
        held_out_topics: Topics excluded from training.
        peak_layer: Layer index to evaluate at.

    Returns:
        Dict with keys train_accuracy, test_accuracy, held_out_topics.
    """
    X = acts_by_layer[peak_layer]
    label_col = _label_column(labels_df)
    y = labels_df[label_col].astype(int).values
    held_mask = labels_df["topic"].isin(held_out_topics).values

    X_train, y_train = X[~held_mask], y[~held_mask]
    X_test, y_test = X[held_mask], y[held_mask]

    clf = LogisticRegression(C=0.1, max_iter=1000).fit(X_train, y_train)
    return {
        "train_accuracy": float(clf.score(X_train, y_train)),
        "test_accuracy": float(clf.score(X_test, y_test)),
        "held_out_topics": list(held_out_topics),
    }


def run_all_holdouts(
    acts_by_layer: dict[int, np.ndarray],
    labels_df: pd.DataFrame,
    peak_layer: int,
) -> pd.DataFrame:
    """Leave-one-topic-out sweep at the peak layer.

    Args:
        acts_by_layer: Dict from probes.load_activations.
        labels_df: As in cross_topic_eval.
        peak_layer: Layer index to evaluate at.

    Returns:
        DataFrame with columns [held_out_topic, train_acc, test_acc].
    """
    rows = []
    for topic in sorted(labels_df["topic"].unique()):
        r = cross_topic_eval(acts_by_layer, labels_df, [topic], peak_layer)
        rows.append({
            "held_out_topic": topic,
            "train_acc": r["train_accuracy"],
            "test_acc": r["test_accuracy"],
        })
    return pd.DataFrame(rows)


def plot_generalization(
    results_df: pd.DataFrame,
    save_path: str = "figures/generalization_by_topic.png",
) -> None:
    """Bar chart of held-out test accuracy per topic.

    Args:
        results_df: Output of run_all_holdouts.
        save_path: Output PNG path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = results_df.sort_values("test_acc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    ax.barh(df["held_out_topic"], df["test_acc"])
    ax.axvline(0.5, linestyle="--", color="gray", label="chance")
    ax.set_xlabel("Test accuracy (held-out topic)")
    ax.set_title("Cross-topic generalisation of abstention probe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
