"""Shared utilities for analysis scripts."""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl


# =============================================================================
# Constants
# =============================================================================

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

# Key columns for baseline grouping and batched parquet reads
BL_KEY_COLS = ["model.hidden_dim", "model.gamma", "data.noise_std", "model.model_seed"]
BATCH_KEY_COLS = BL_KEY_COLS[:3]


# =============================================================================
# Statistics
# =============================================================================


def mean_centered_spread(
    curves: np.ndarray, mean: np.ndarray, coverage: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean-centered spread bands capturing `coverage` of runs.

    For each half of the distribution (above/below the mean), captures
    `coverage` of runs in that half. This centers the band on the mean
    rather than the median, giving asymmetric bands for skewed distributions.
    """
    n_runs, n_steps = curves.shape
    tail = 1.0 - coverage

    # Fraction of runs at or below the mean at each step
    f_below = (curves <= mean[np.newaxis, :]).sum(axis=0) / n_runs

    # Target quantiles (vary per step to center on mean)
    q_lo = tail * f_below
    q_hi = tail * f_below + coverage

    # Sort once and interpolate per-step quantiles
    sorted_curves = np.sort(curves, axis=0)
    step_idx = np.arange(n_steps)

    def _interp(q):
        idx_f = q * (n_runs - 1)
        lo = np.clip(np.floor(idx_f).astype(int), 0, n_runs - 2)
        frac = idx_f - lo
        return (
            sorted_curves[lo, step_idx] * (1 - frac)
            + sorted_curves[lo + 1, step_idx] * frac
        )

    return _interp(q_lo), _interp(q_hi)


def compute_ci(
    curves: np.ndarray, z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and z-based confidence interval for (n_seeds, ...) curves.

    Works for both 2D (n, steps) and 3D (n, steps, svs) arrays.
    Returns (mean, ci_lo, ci_hi) with the same shape as curves[0].
    """
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    if n == 1:
        return mean, mean, mean
    sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, mean - z * sem, mean + z * sem


# =============================================================================
# Data Loading
# =============================================================================


def extract_curves(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract a list-of-lists column as a (n_rows, n_steps) numpy array."""
    return np.vstack(df[col].to_list())


def extract_sv_3d(df: pl.DataFrame, col: str, n_rows: int) -> np.ndarray:
    """Extract SV column as a (n_rows, n_steps, n_svs) numpy array."""
    return np.stack([np.stack(df[col][i]) for i in range(n_rows)])


def data_fingerprint(data_root: Path, subdirs: list[str]) -> str:
    """MD5 hash of mtime+size of parquet files — invalidates cache on data change."""
    h = hashlib.md5()
    for name in subdirs:
        p = data_root / name / "results.parquet"
        if p.exists():
            stat = p.stat()
            h.update(f"{p}:{stat.st_mtime}:{stat.st_size}".encode())
    return h.hexdigest()[:12]


def build_filter(key_cols: list[str], values: tuple) -> pl.Expr:
    """Build a polars filter expression matching key columns to values."""
    expr = pl.lit(True)
    for col, val in zip(key_cols, values):
        expr = expr & (pl.col(col) == val)
    return expr


def sort_parquet(
    exp_config: dict, key_cols: list[str] = BL_KEY_COLS,
) -> None:
    """Sort parquet files by key columns for efficient predicate pushdown.

    Parquet stores per-row-group column statistics (min/max). When rows are
    sorted by the columns used in filter predicates, each row group has tight
    value ranges, allowing polars to skip irrelevant row groups entirely.
    Without sorting, every row group spans the full value range and nothing
    can be skipped — each filtered read must scan the entire file.
    """
    for subdir in [exp_config["baseline_subdir"], exp_config["sgd_subdir"]]:
        path = exp_config["base_path"] / subdir / "results.parquet"
        if not path.exists():
            print(f"  Skipping {path} (not found)")
            continue

        lf = pl.scan_parquet(path)
        schema_cols = lf.collect_schema().names()
        sort_cols = [c for c in key_cols if c in schema_cols]
        if "training.batch_size" in schema_cols:
            sort_cols.append("training.batch_size")

        print(f"  Sorting {path} by {sort_cols}...")
        tmp_path = path.with_suffix(".parquet.tmp")
        df = lf.sort(sort_cols).collect(engine="streaming")
        df.write_parquet(tmp_path, row_group_size=50_000)
        tmp_path.replace(path)
        print(f"  Rewritten ({len(df):,} rows, {path.stat().st_size / 1e6:.0f} MB)")


# =============================================================================
# Plotting Helpers
# =============================================================================


def fmt_seeds(n) -> str:
    """Format seed count for titles, e.g. 10000 -> '10,000'."""
    return f"{n:,}" if isinstance(n, int) else str(n)


def suptitle_params(
    exp_config: dict, gamma: float, noise: float,
) -> str:
    """Build pipe-separated shared parameter string for figure suptitles."""
    baseline_bs = exp_config["baseline_batch_size"]
    regime = "Online (infinite data)" if baseline_bs is not None else "Fixed train set (N=500)"

    parts = [
        regime,
        f"{GAMMA_NAMES[gamma]} initialisation (\u03b3 = {gamma})",
        f"Label noise std = {noise}",
    ]
    return " | ".join(parts)


def pp_label(i: int, j: int) -> str:
    """LaTeX label for partial product W_j ... W_i."""
    if i == j:
        return f"$W_{{{j}}}$"
    layers = " ".join(f"W_{{{k}}}" for k in range(j, i - 1, -1))
    return f"${layers}$"


def sync_ylims(axes: np.ndarray) -> None:
    """Set consistent y-axis limits across columns for each row."""
    n_rows, n_cols = axes.shape
    for r in range(n_rows):
        ymin = min(axes[r, c].get_ylim()[0] for c in range(n_cols))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(n_cols))
        for c in range(n_cols):
            axes[r, c].set_ylim(ymin, ymax)
