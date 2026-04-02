"""Shared constants and helpers for LR sweep analysis scripts."""

from pathlib import Path

import numpy as np

DATA_PATH = Path("outputs/lr_sweep_online/results.parquet")
FIGURES_PATH = Path("figures/lr_sweep_online")

LARGE_BS = 500
SMALL_BS_LIST = [50, 5, 1]

# Per-batch-size LR grids (6 log-spaced from 1e-4 to max stable)
LR_GRIDS = {
    50: np.logspace(np.log10(0.0001), np.log10(0.0041), 6),
    5:  np.logspace(np.log10(0.0001), np.log10(0.003), 6),
    1:  np.logspace(np.log10(0.0001), np.log10(0.001), 6),
}


def closest_lr(target: float, available: list[float]) -> float:
    """Find the available LR closest to the target."""
    return min(available, key=lambda x: abs(x - target))


def bs_label(bs: int, online: bool) -> str:
    """Human-readable batch size label."""
    if not online and bs == LARGE_BS:
        return "GD"
    return f"Batch size {bs}"
