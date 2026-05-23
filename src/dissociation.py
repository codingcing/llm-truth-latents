"""Characterise the knowledge/behaviour dissociation.

Notebook 02 found Llama 3.1 8B represents answerability almost perfectly (~97%
linear probe) yet acts on it far less reliably (~74%). This module supports the
follow-up question: in the failure cases — unanswerable questions the model
answered anyway — what is different about the activations?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (gold answerability x judged behaviour). unanswerable_answered is the failure
# case of interest: the model "knew" the question was unanswerable but answered.
QUADRANTS = [
    "answerable_answered",
    "answerable_abstained",
    "unanswerable_answered",
    "unanswerable_abstained",
]


def assign_quadrants(labels_df: pd.DataFrame) -> pd.Series:
    """Tag each row by (gold answerability x judged behaviour).

    Args:
        labels_df: DataFrame with a boolean `answerable` column and a
            `judge_label` column whose values are "answered" / "abstained".

    Returns:
        Series of quadrant strings (values in QUADRANTS), aligned to labels_df.
    """
    answerable = labels_df["answerable"].astype(bool).to_numpy()
    answered = (labels_df["judge_label"] == "answered").to_numpy()
    quad = np.where(
        answerable,
        np.where(answered, "answerable_answered", "answerable_abstained"),
        np.where(answered, "unanswerable_answered", "unanswerable_abstained"),
    )
    return pd.Series(quad, index=labels_df.index, name="quadrant")


def project_onto_direction(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Scalar projection of each activation onto a direction.

    The direction is normalised internally, so projections are in
    activation-norm units regardless of how `direction` was scaled.

    Args:
        acts: (N, D) activations from a single layer.
        direction: (D,) direction vector (e.g. a difference-of-means vector).

    Returns:
        (N,) array of projection values — higher means further along `direction`.
    """
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 0 else direction
    return acts @ unit
