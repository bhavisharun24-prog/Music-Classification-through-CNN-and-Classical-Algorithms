"""Reusable plotting helpers for experiment outputs.

The training and evaluation modules contain the project-specific plotting
implementations. This module provides a small public surface for notebooks or
future scripts without duplicating that logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .evaluation import (
    plot_confusion_matrix,
    plot_grouped_bars,
    plot_history,
    plot_pca_projection,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_grouped_bars",
    "plot_history",
    "plot_pca_projection",
    "save_genre_distribution",
]


def save_genre_distribution(
    labels: Sequence[int],
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    """Save a simple genre-count bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = pd.Series(labels).value_counts().reindex(range(len(class_names)), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(class_names, counts.to_numpy())
    ax.set_xlabel("Genre")
    ax.set_ylabel("Songs")
    ax.set_title("Genre distribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
