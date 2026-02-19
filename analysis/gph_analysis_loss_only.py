"""
GPH Sweep Analysis

Analyzes results from GPH experiment sweeps (offline or online) with:
- Parquet-based data loading via polars
- Caching of computed statistics for fast plot iteration
- Two plot types: loss ratio with significance testing, loss variability with spread bands
- Parallel figure generation via multiprocessing

Usage (run from the project root):

    python analysis/gph_analysis_loss_only.py offline
    python analysis/gph_analysis_loss_only.py online

Options:
    --recompute       Force recompute statistics (ignore cache)
    --sort-parquet    Sort parquet files by key columns for efficient predicate
                      pushdown (one-time preprocessing step), then exit

Expects sweep outputs in outputs/gph/gph_offline_loss/ or outputs/gph/gph_online_loss/.
Saves figures to figures/gph_offline_loss/ or figures/gph_online_loss/.
Caches computed statistics in .analysis_cache/gph_offline_loss.pkl or .analysis_cache/gph_online_loss.pkl.
"""

import argparse
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as scipy_stats


# =============================================================================
# Configuration
# =============================================================================

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

BL_KEY_COLS = ["model.hidden_dim", "model.gamma", "data.noise_std", "model.model_seed"]
BATCH_KEY_COLS = BL_KEY_COLS[:3]  # columns used for batched parquet reads

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph/gph_offline_loss"),
        "cache_path": Path(".analysis_cache/gph_offline_loss.pkl"),
        "figures_path": Path("figures/gph_offline_loss"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/gph/gph_online_loss"),
        "cache_path": Path(".analysis_cache/gph_online_loss.pkl"),
        "figures_path": Path("figures/gph_online_loss"),
        "baseline_subdir": "large_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
        "regime_label": "Online (infinite data)",
    },
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class BaselineStats:
    """Statistics for a baseline (GD or large-batch) configuration."""

    steps: np.ndarray
    loss: np.ndarray
    var: np.ndarray | None  # None for deterministic (offline) baseline
    n: int
    min: np.ndarray | None
    max: np.ndarray | None
    spread_lo: np.ndarray | None  # mean-centered 90% band
    spread_hi: np.ndarray | None


@dataclass
class ConfigStats:
    """Precomputed statistics for a single (width, gamma, noise, seed, batch_size) config.

    All plotting quantities are precomputed during the statistics phase so the
    plot loop does no redundant numpy/scipy work.
    """

    steps: np.ndarray
    # Baseline
    baseline_loss: np.ndarray
    baseline_n: int
    baseline_min: np.ndarray | None
    baseline_max: np.ndarray | None
    baseline_spread_lo: np.ndarray | None
    baseline_spread_hi: np.ndarray | None
    # SGD summary
    sgd_mean: np.ndarray
    sgd_min: np.ndarray
    sgd_max: np.ndarray
    sgd_spread_lo: np.ndarray
    sgd_spread_hi: np.ndarray
    sgd_log_std: np.ndarray
    n_runs: int
    # Precomputed for loss ratio plot
    sgd_ci_lo: np.ndarray  # log-space 95% CI lower bound
    sgd_ci_hi: np.ndarray  # log-space 95% CI upper bound
    ratio: np.ndarray  # sgd_mean / baseline_loss
    ratio_ci_lo: np.ndarray
    ratio_ci_hi: np.ndarray
    sig_mask: np.ndarray  # bool: baseline significantly < SGD at 95%


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_curves(df: pl.DataFrame) -> np.ndarray:
    """Extract loss curves as a (n_runs, n_steps) numpy array."""
    return np.vstack(df["test_loss"].to_list())


def _mean_centered_spread(
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


def _welch_t_crit(
    se_a: np.ndarray, n_a: int, se_b: np.ndarray, n_b: int,
) -> np.ndarray:
    """Welch-Satterthwaite t critical value (two-tailed 95%), numerically stable.

    se_a, se_b are variance/n (squared standard error of the mean).
    When both SEs underflow to zero, returns 1.96 (normal approximation,
    appropriate since both means are precisely known).
    """
    se_sum = se_a + se_b
    ws_numer = se_sum**2
    ws_denom = se_a**2 / max(n_a - 1, 1) + se_b**2 / max(n_b - 1, 1)
    # When ws_denom underflows to 0, SEs are negligible → large df (normal approx)
    nonzero = ws_denom > 0
    df = np.where(nonzero, ws_numer / np.where(nonzero, ws_denom, 1.0), 1e6)
    df = np.maximum(df, 1.0)
    return scipy_stats.t.ppf(0.975, df=df)


def _compute_baseline_stats(subset: pl.DataFrame) -> BaselineStats | None:
    """Compute baseline loss statistics from a pre-filtered group.

    Offline (full_batch): single deterministic GD run.
    Online (large_batch): mean/variance/min/max/spread over batch seeds.
    """
    if len(subset) == 0:
        return None

    steps = np.array(subset["step"][0])
    curves = _extract_curves(subset)
    n = len(curves)

    if n == 1:
        return BaselineStats(steps, curves[0], None, 1, None, None, None, None)

    mean = curves.mean(axis=0)
    spread_lo, spread_hi = _mean_centered_spread(curves, mean)

    return BaselineStats(
        steps=steps,
        loss=mean,
        var=curves.var(axis=0, ddof=1),
        n=n,
        min=curves.min(axis=0),
        max=curves.max(axis=0),
        spread_lo=spread_lo,
        spread_hi=spread_hi,
    )


def _compute_sgd_config_stats(
    subset: pl.DataFrame, bl: BaselineStats,
) -> ConfigStats | None:
    """Compute SGD statistics and precompute all plotting quantities."""
    if len(subset) == 0:
        return None

    curves = _extract_curves(subset)
    n = len(curves)

    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1) if n > 1 else np.zeros(curves.shape[1])
    spread_lo, spread_hi = (
        _mean_centered_spread(curves, mean) if n > 1 else (mean, mean)
    )

    # Log-space statistics
    log_std = (
        np.log(np.maximum(curves, 1e-300)).std(axis=0, ddof=1)
        if n > 1
        else np.zeros(curves.shape[1])
    )

    # --- Precompute plotting quantities ---

    t_val = scipy_stats.t.ppf(0.975, df=max(n - 1, 1))
    sem = np.sqrt(var / n)

    # Log-space CI for the arithmetic mean (delta method on log(X̄)):
    # Var(log X̄) ≈ SEM²/μ², so CI = mean ×/÷ exp(t × SEM/mean)
    relative_sem = sem / np.maximum(mean, 1e-30)
    ci_factor = np.exp(t_val * relative_sem)
    sgd_ci_lo = mean / ci_factor
    sgd_ci_hi = mean * ci_factor

    # Linear-space CI bounds (used for significance test and offline ratio)
    linear_lo = mean - t_val * sem
    linear_hi = mean + t_val * sem

    # Loss ratio E(L_SGD) / L_baseline and its CI
    ratio = mean / bl.loss

    if bl.var is None:
        # Deterministic baseline: CI transforms linearly
        ratio_ci_lo = linear_lo / bl.loss
        ratio_ci_hi = linear_hi / bl.loss
        sig_mask = bl.loss < linear_lo
    else:
        # Delta method: Var(Y/X) ≈ (Y/X)² × (Var(Y)/(n_Y·Y²) + Var(X)/(n_X·X²))
        safe_sgd = np.maximum(mean, 1e-30)
        safe_bl = np.maximum(bl.loss, 1e-30)
        rel_var = var / (n * safe_sgd**2) + bl.var / (bl.n * safe_bl**2)
        se_ratio = ratio * np.sqrt(rel_var)

        se_bl = bl.var / bl.n
        se_sgd = var / n
        t_crit = _welch_t_crit(se_bl, bl.n, se_sgd, n)

        ratio_ci_lo = ratio - t_crit * se_ratio
        ratio_ci_hi = ratio + t_crit * se_ratio

        # Welch's two-sample test: E[SGD] - E[baseline] > 0 at 95% confidence
        diff = mean - bl.loss
        se_diff = np.sqrt(se_bl + se_sgd)
        sig_mask = diff > t_crit * se_diff

    return ConfigStats(
        steps=bl.steps,
        baseline_loss=bl.loss,
        baseline_n=bl.n,
        baseline_min=bl.min,
        baseline_max=bl.max,
        baseline_spread_lo=bl.spread_lo,
        baseline_spread_hi=bl.spread_hi,
        sgd_mean=mean,
        sgd_min=curves.min(axis=0),
        sgd_max=curves.max(axis=0),
        sgd_spread_lo=spread_lo,
        sgd_spread_hi=spread_hi,
        sgd_log_std=log_std,
        n_runs=n,
        sgd_ci_lo=sgd_ci_lo,
        sgd_ci_hi=sgd_ci_hi,
        ratio=ratio,
        ratio_ci_lo=ratio_ci_lo,
        ratio_ci_hi=ratio_ci_hi,
        sig_mask=sig_mask,
    )


# =============================================================================
# Statistics Computation
# =============================================================================


def _build_filter(key_cols: list[str], values: tuple) -> pl.Expr:
    """Build a polars filter expression matching key columns to values."""
    expr = pl.lit(True)
    for col, val in zip(key_cols, values):
        expr = expr & (pl.col(col) == val)
    return expr


def compute_all_stats(exp_config: dict) -> dict[tuple, ConfigStats]:
    baseline_dir = exp_config["base_path"] / exp_config["baseline_subdir"]
    sgd_dir = exp_config["base_path"] / exp_config["sgd_subdir"]

    # Phase 1: Baselines — small enough to load fully
    print(f"Loading baselines from {baseline_dir}...")
    baseline_df = (
        pl.scan_parquet(baseline_dir / "results.parquet")
        .select(BL_KEY_COLS + ["step", "test_loss"])
        .collect()
    )
    baseline_groups = baseline_df.partition_by(BL_KEY_COLS, as_dict=True)

    print(f"Computing baselines ({len(baseline_groups)} configs)...")
    baselines: dict[tuple, BaselineStats] = {}
    for key, group_df in baseline_groups.items():
        result = _compute_baseline_stats(group_df)
        if result is not None:
            baselines[key] = result
    del baseline_df, baseline_groups
    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — batch by (width, gamma, noise) to reduce parquet scans.
    # Each batch loads all model seeds and batch sizes for one (width, gamma, noise)
    # combination, keeping memory bounded while reducing I/O.
    # For best performance, pre-sort the parquet file with --sort-parquet so that
    # polars can skip irrelevant row groups via predicate pushdown.
    sgd_lf = pl.scan_parquet(sgd_dir / "results.parquet")
    # "step" is omitted — identical across runs, already stored in BaselineStats
    sgd_select = BL_KEY_COLS + ["training.batch_size", "test_loss"]

    key_batches: dict[tuple, list[tuple]] = defaultdict(list)
    for bl_key in sorted(baselines):
        key_batches[bl_key[:3]].append(bl_key)

    n_batches = len(key_batches)
    total_configs = len(baselines)
    print(f"Computing SGD statistics ({n_batches} batches, {total_configs} baseline groups)...")

    stats: dict[tuple, ConfigStats] = {}
    completed = 0
    for batch_key, batch_bl_keys in key_batches.items():
        chunk = (
            sgd_lf.filter(_build_filter(BATCH_KEY_COLS, batch_key))
            .select(sgd_select)
            .collect()
        )
        sub_groups = chunk.partition_by(
            ["model.model_seed", "training.batch_size"], as_dict=True,
        )
        del chunk

        # Iterate sub-groups and look up matching baseline directly
        for (model_seed, batch_size), group_df in sub_groups.items():
            bl_key = batch_key + (model_seed,)
            bl = baselines.get(bl_key)
            if bl is None:
                continue
            key = bl_key + (batch_size,)
            result = _compute_sgd_config_stats(group_df, bl)
            if result is not None:
                stats[key] = result

        completed += len(batch_bl_keys)
        print(
            f"\r  SGD stats: {completed}/{total_configs}"
            f" ({100 * completed / total_configs:.0f}%)",
            end="",
            flush=True,
        )
        del sub_groups

    print(f"\n  Computed {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def save_cache(stats: dict[tuple, ConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {key: vars(cs) for key, cs in stats.items()}
    with open(path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        return {key: ConfigStats(**d) for key, d in cache_data.items()}
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    exp_config: dict, force_recompute: bool = False,
) -> dict[tuple, ConfigStats]:
    cache_path = exp_config["cache_path"]
    if not force_recompute:
        stats = load_cache(cache_path)
        if stats is not None:
            print(f"Loaded {len(stats)} configurations from cache")
            return stats

    stats = compute_all_stats(exp_config)
    save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting
# =============================================================================


def _fmt_seeds(n) -> str:
    """Format seed count for titles, e.g. 10000 → '10,000'."""
    if isinstance(n, int):
        return f"{n:,}"
    return str(n)


def _suptitle_params(
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


def _get_n_seeds(stats: dict[tuple, ConfigStats]):
    """Look up the number of batch seeds from any config."""
    sample_key = next(iter(stats), None)
    return stats[sample_key].n_runs if sample_key else "?"


def plot_loss_ratio(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    gamma: float,
    noise: float,
    batch_size: int,
    widths: list,
    model_seeds: list,
) -> plt.Figure:
    """Loss ratio analysis: model seeds as columns, widths stacked vertically.

    Per width block (2 rows): loss curves with significance shading, then ratio with CI.
    Columns: one per model seed.
    """
    baseline_bs = exp_config["baseline_batch_size"]

    # Width-independent labels
    if baseline_bs is not None:
        ratio_ylabel = f"$E(L_{{B={batch_size}}}) \\,/\\, E(L_{{B={baseline_bs}}})$"
        bl_legend = f"$E(L_{{B={baseline_bs}}})$"
        sgd_legend = f"$E(L_{{B={batch_size}}})$"
        sig_label = f"$E(L_{{B={baseline_bs}}}) < E(L_{{B={batch_size}}})$ (p<0.05)"
    else:
        ratio_ylabel = "$E(L_{\\mathrm{SGD}}) \\,/\\, L_{\\mathrm{GD}}$"
        bl_legend = "$L_{\\mathrm{GD}}$"
        sgd_legend = "$E(L_{\\mathrm{SGD}})$"
        sig_label = "$L_{\\mathrm{GD}} < E(L_{\\mathrm{SGD}})$ (p<0.05)"

    n_cols = len(model_seeds)
    n_widths = len(widths)
    fig, axes = plt.subplots(
        2 * n_widths, n_cols, figsize=(5 * n_cols, 2.5 * 2 * n_widths), squeeze=False,
    )

    for w_idx, width in enumerate(widths):
        row_base = w_idx * 2
        is_top = w_idx == 0
        is_bottom = w_idx == n_widths - 1

        for col, model_seed in enumerate(model_seeds):
            ax_loss = axes[row_base, col]
            ax_ratio = axes[row_base + 1, col]

            key = (width, gamma, noise, model_seed, batch_size)

            if key not in stats:
                if is_top:
                    ax_loss.set_title(f"Model Seed {model_seed}")
                if col == 0:
                    ax_loss.set_ylabel(f"Width {width}\nTest loss")
                    ax_ratio.set_ylabel(f"Width {width}\n{ratio_ylabel}")
                if is_bottom:
                    ax_loss.set_xlabel("Training step")
                    ax_ratio.set_xlabel("Training step")
                continue

            s = stats[key]

            # === Loss row (top) ===
            ax_loss.plot(
                s.steps, s.baseline_loss,
                label=bl_legend, color="C0", linewidth=1.5,
            )
            ax_loss.plot(
                s.steps, s.sgd_mean,
                label=sgd_legend, color="C1", linewidth=1.5,
            )
            ax_loss.fill_between(
                s.steps, s.sgd_ci_lo, s.sgd_ci_hi,
                alpha=0.3, color="C1", label=f"{sgd_legend} 95% CI",
            )
            ax_loss.fill_between(
                s.steps, 0, 1,
                where=s.sig_mask, alpha=0.4, color="darkgreen",
                transform=ax_loss.get_xaxis_transform(), label=sig_label,
            )
            ax_loss.set_yscale("log")
            if is_top:
                ax_loss.set_title(f"Model Seed {model_seed}")
            if col == 0:
                ax_loss.set_ylabel(f"Width {width}\nTest loss")
            if is_bottom:
                ax_loss.set_xlabel("Training step")
            ax_loss.legend(loc="upper right", fontsize=6)

            # === Ratio row (bottom) ===
            ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
            ax_ratio.plot(
                s.steps, s.ratio, color="C1", linewidth=1.5,
            )
            ax_ratio.fill_between(
                s.steps, s.ratio_ci_lo, s.ratio_ci_hi,
                alpha=0.3, color="C1", label="95% CI",
            )
            if col == 0:
                ax_ratio.set_ylabel(f"Width {width}\n{ratio_ylabel}")
            if is_bottom:
                ax_ratio.set_xlabel("Training step")
            ax_ratio.legend(loc="upper right", fontsize=7)

    n_batch_seeds = _get_n_seeds(stats)
    if baseline_bs is not None:
        prefix = (
            f"$E(L_{{B={baseline_bs}}})$ vs "
            f"$E(L_{{B={batch_size}}})$ over "
            f"{_fmt_seeds(n_batch_seeds)} batch seeds"
        )
    else:
        prefix = (
            f"$L_{{\\mathrm{{GD}}}}$ vs "
            f"$E(L_{{\\mathrm{{SGD}}}})$ over "
            f"{_fmt_seeds(n_batch_seeds)} batch partitions | batch size = {batch_size}"
        )
    params = _suptitle_params(exp_config, gamma, noise)
    fig.suptitle(
        f"{prefix} | {params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_loss_variability(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    gamma: float,
    noise: float,
    batch_size: int,
    widths: list,
    model_seeds: list,
) -> plt.Figure:
    """Loss variability analysis: model seeds as columns, widths stacked vertically.

    Per width block (2 rows): σ(log L), then loss with P5–P95 and min–max bands.
    Columns: one per model seed.
    """
    baseline_bs = exp_config["baseline_batch_size"]

    # Width-independent labels
    if baseline_bs is not None:
        bl_legend = f"$E(L_{{B={baseline_bs}}})$"
        sgd_legend = f"$E(L_{{B={batch_size}}})$"
        cv_legend = sgd_legend
    else:
        bl_legend = "$L_{\\mathrm{GD}}$"
        sgd_legend = "$E(L_{\\mathrm{SGD}})$"
        cv_legend = sgd_legend

    n_cols = len(model_seeds)
    n_widths = len(widths)
    fig, axes = plt.subplots(
        2 * n_widths, n_cols, figsize=(5 * n_cols, 2.5 * 2 * n_widths), squeeze=False,
    )

    for w_idx, width in enumerate(widths):
        row_base = w_idx * 2
        is_top = w_idx == 0
        is_bottom = w_idx == n_widths - 1

        for col, model_seed in enumerate(model_seeds):
            ax_cv = axes[row_base, col]
            ax_loss = axes[row_base + 1, col]

            key = (width, gamma, noise, model_seed, batch_size)

            if key not in stats:
                if is_top:
                    ax_cv.set_title(f"Model Seed {model_seed}")
                if col == 0:
                    ax_cv.set_ylabel(f"Width {width}\n" + r"$\sigma(\log L)$")
                    ax_loss.set_ylabel(f"Width {width}\nTest loss")
                if is_bottom:
                    ax_cv.set_xlabel("Training step")
                    ax_loss.set_xlabel("Training step")
                continue

            s = stats[key]

            # === σ(log L) row ===
            ax_cv.plot(
                s.steps, s.sgd_log_std,
                color="C1", linewidth=1.5, label=cv_legend,
            )
            if is_top:
                ax_cv.set_title(f"Model Seed {model_seed}")
            if col == 0:
                ax_cv.set_ylabel(f"Width {width}\n" + r"$\sigma(\log L)$")
            if is_bottom:
                ax_cv.set_xlabel("Training step")
            ax_cv.legend(loc="upper right", fontsize=7)

            # === Loss with spread bands row ===
            ax_loss.plot(
                s.steps, s.baseline_loss,
                label=bl_legend, color="C0", linewidth=1.5,
            )
            # Baseline bands (online experiment only)
            if s.baseline_min is not None:
                ax_loss.fill_between(
                    s.steps, s.baseline_min, s.baseline_max,
                    alpha=0.25, color="C0", label="min\u2013max",
                )
                ax_loss.fill_between(
                    s.steps, s.baseline_spread_lo, s.baseline_spread_hi,
                    alpha=0.3, color="C0", label="90% around mean",
                )
            ax_loss.plot(
                s.steps, s.sgd_mean,
                label=sgd_legend, color="C1", linewidth=1.5,
            )
            ax_loss.fill_between(
                s.steps, s.sgd_min, s.sgd_max,
                alpha=0.25, color="C1", label="min\u2013max",
            )
            ax_loss.fill_between(
                s.steps, s.sgd_spread_lo, s.sgd_spread_hi,
                alpha=0.3, color="C1", label="90% around mean",
            )
            ax_loss.set_yscale("log")
            if col == 0:
                ax_loss.set_ylabel(f"Width {width}\nTest loss")
            if is_bottom:
                ax_loss.set_xlabel("Training step")
            ax_loss.legend(loc="upper right", fontsize=6)

    n_batch_seeds = _get_n_seeds(stats)
    if baseline_bs is not None:
        prefix = (
            f"$L_{{B={baseline_bs}}}$ and "
            f"$L_{{B={batch_size}}}$ across "
            f"{_fmt_seeds(n_batch_seeds)} batch seeds"
        )
    else:
        prefix = (
            f"$L_{{\\mathrm{{SGD}}}}$ across "
            f"{_fmt_seeds(n_batch_seeds)} batch partitions | batch size = {batch_size}"
        )
    params = _suptitle_params(exp_config, gamma, noise)
    fig.suptitle(
        f"{prefix} | {params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Parallel Plot Generation
# =============================================================================

# Module-level state for multiprocessing workers (set by _init_plot_worker)
_worker_ctx: dict = {}


def _init_plot_worker(stats: dict, exp_config: dict) -> None:
    """Initializer for plot worker processes. Called once per worker."""
    _worker_ctx["stats"] = stats
    _worker_ctx["exp"] = exp_config


def _run_plot_task(task: tuple) -> None:
    """Worker function: generate one figure and save to disk."""
    plot_type, gamma, noise, batch_size, widths, model_seeds, filename = task
    stats = _worker_ctx["stats"]
    exp = _worker_ctx["exp"]

    plot_fn = plot_loss_ratio if plot_type == "ratio" else plot_loss_variability
    fig = plot_fn(exp, stats, gamma, noise, batch_size, widths, model_seeds)
    fig.savefig(exp["figures_path"] / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(exp_config: dict, stats: dict[tuple, ConfigStats]) -> None:
    figures_path = exp_config["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)

    # Derive parameter values from stats keys: (width, gamma, noise, model_seed, batch_size)
    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    # Build task list
    tasks = []
    for gamma in gammas:
        for noise in noise_levels:
            for batch_size in batch_sizes:
                name = f"g{gamma}_noise{noise}_b{batch_size}"
                common = (gamma, noise, batch_size, widths, model_seeds)
                tasks.append(("ratio", *common, f"ratio_{name}"))
                tasks.append(("variability", *common, f"variability_{name}"))

    n_workers = min(os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} figures across {n_workers} workers...")

    with Pool(n_workers, initializer=_init_plot_worker, initargs=(stats, exp_config)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_plot_task, tasks), 1):
            print(
                f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="",
                flush=True,
            )

    print(f"\nAll plots saved to {figures_path}/")


# =============================================================================
# Parquet Sorting
# =============================================================================


def sort_parquet(exp_config: dict) -> None:
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
        sort_cols = [c for c in BL_KEY_COLS if c in schema_cols]
        if "training.batch_size" in schema_cols:
            sort_cols.append("training.batch_size")

        print(f"  Sorting {path} by {sort_cols}...")
        df = lf.sort(sort_cols).collect(engine="streaming")
        df.write_parquet(path, row_group_size=50_000)
        print(f"  Rewritten ({len(df):,} rows, {path.stat().st_size / 1e6:.0f} MB)")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze GPH sweep results")
    parser.add_argument(
        "experiment",
        choices=["offline", "online"],
        help="Which experiment to analyze",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    parser.add_argument(
        "--sort-parquet",
        action="store_true",
        help="Sort parquet files by key columns for efficient reads, then exit",
    )
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]

    print(f"Experiment: {args.experiment}")
    print(f"Data: {exp['base_path']}")

    if args.sort_parquet:
        sort_parquet(exp)
        print("Done! Re-run without --sort-parquet to analyze.")
        return

    stats = get_stats(exp, force_recompute=args.recompute)
    generate_all_plots(exp, stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
