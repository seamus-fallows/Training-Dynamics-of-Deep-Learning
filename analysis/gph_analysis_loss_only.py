"""
GPH Sweep Analysis

Analyzes results from GPH experiment sweeps (offline or online) with:
- Parquet-based data loading via polars
- Caching of computed statistics for fast plot iteration
- Two plot types: loss ratio with significance testing, loss variability with spread bands

Usage (run from the project root):

    python analysis/gph_analysis_loss_only.py offline
    python analysis/gph_analysis_loss_only.py online

Options:
    --recompute    Force recompute statistics (ignore cache)

Expects sweep outputs in outputs/gph/gph_offline_loss/ or outputs/gph/gph_online_loss/.
Saves figures to figures/gph_offline_loss/ or figures/gph_online_loss/.
Caches computed statistics in cache/gph_offline_loss.pkl or cache/gph_online_loss.pkl.
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# =============================================================================
# Configuration
# =============================================================================

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

# Histogram for density bands: log10-spaced bins covering plausible loss range
LOG10_LOSS_RANGE = (-13, 4)
N_HIST_BINS = 200
HIST_EDGES = np.linspace(*LOG10_LOSS_RANGE, N_HIST_BINS + 1)

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph/gph_offline_loss"),
        "cache_path": Path("cache/gph_offline_loss.pkl"),
        "figures_path": Path("figures/gph_offline_loss"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/gph/gph_online_loss"),
        "cache_path": Path("cache/gph_online_loss.pkl"),
        "figures_path": Path("figures/gph_online_loss"),
        "baseline_subdir": "large_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
        "regime_label": "Online (infinite data)",
    },
}

# Set in main()
_exp = None


# =============================================================================
# Data Loading
# =============================================================================


def _scan_sweep(sweep_dir: Path) -> pl.LazyFrame:
    """Return a LazyFrame for a sweep's results.parquet."""
    return pl.scan_parquet(sweep_dir / "results.parquet")


def _extract_curves(df: pl.DataFrame) -> np.ndarray:
    """Extract loss curves as a (n_runs, n_steps) numpy array."""
    n = len(df)
    flat = df["test_loss"].explode().to_numpy()
    return flat.reshape(n, len(flat) // n)


def _compute_histogram(curves: np.ndarray) -> np.ndarray:
    """Compute per-step loss histogram in log10 space.

    Args:
        curves: (n_runs, n_steps) array of loss values.

    Returns:
        (n_steps, N_HIST_BINS) histogram counts.
    """
    n_runs, n_steps = curves.shape
    log10_curves = np.log10(np.maximum(curves, 10.0 ** LOG10_LOSS_RANGE[0]))
    bin_idx = np.digitize(log10_curves, HIST_EDGES) - 1
    np.clip(bin_idx, 0, N_HIST_BINS - 1, out=bin_idx)
    step_idx = np.broadcast_to(np.arange(n_steps), (n_runs, n_steps))
    flat_idx = (step_idx * N_HIST_BINS + bin_idx).ravel()
    counts = np.bincount(flat_idx, minlength=n_steps * N_HIST_BINS)
    return counts[: n_steps * N_HIST_BINS].reshape(n_steps, N_HIST_BINS).astype(np.int32)


def _compute_baseline_stats(subset: pl.DataFrame) -> tuple | None:
    """Compute baseline loss statistics from a pre-filtered group.

    Offline (full_batch): single deterministic GD run.
        Returns (steps, loss, None, 1, None, None, None, None).
    Online (large_batch): mean/variance/min/max/histogram over batch seeds.
        Returns (steps, mean, var, n, min, max, hist_counts, hist_edges).
    """
    if len(subset) == 0:
        return None

    steps = np.array(subset["step"][0])
    curves = _extract_curves(subset)
    n = len(curves)

    if n == 1:
        return steps, curves[0], None, 1, None, None, None, None

    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1)
    min_vals = curves.min(axis=0)
    max_vals = curves.max(axis=0)
    histogram = _compute_histogram(curves)

    return steps, mean, var, n, min_vals, max_vals, histogram, HIST_EDGES


@dataclass
class ConfigStats:
    """Statistics for a single configuration."""

    steps: np.ndarray
    baseline_loss: np.ndarray
    baseline_var: np.ndarray | None  # None for deterministic (offline) baseline
    baseline_n: int
    baseline_min: np.ndarray | None  # None for deterministic (offline) baseline
    baseline_max: np.ndarray | None
    baseline_hist_counts: np.ndarray | None  # (n_steps, N_HIST_BINS) or None
    baseline_hist_edges: np.ndarray | None  # (N_HIST_BINS + 1,) or None
    sgd_mean: np.ndarray
    sgd_lower: np.ndarray
    sgd_upper: np.ndarray
    sgd_min: np.ndarray
    sgd_max: np.ndarray
    sgd_var: np.ndarray
    sgd_log_mean: np.ndarray  # mean of log(loss)
    sgd_log_std: np.ndarray  # std of log(loss), for log-space spread bands
    hist_counts: np.ndarray  # (n_steps, N_HIST_BINS) loss histogram in log10 space
    hist_edges: np.ndarray  # (N_HIST_BINS + 1,) bin edges in log10 space
    n_runs: int


def _compute_sgd_config_stats(
    subset: pl.DataFrame,
    baseline_steps: np.ndarray,
    baseline_loss: np.ndarray,
    baseline_var: np.ndarray | None,
    baseline_n: int,
    baseline_min: np.ndarray | None,
    baseline_max: np.ndarray | None,
    baseline_hist_counts: np.ndarray | None,
    baseline_hist_edges: np.ndarray | None,
) -> ConfigStats | None:
    """Compute SGD statistics from a pre-filtered group."""
    if len(subset) == 0:
        return None

    curves = _extract_curves(subset)
    n = len(curves)

    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1) if n > 1 else np.zeros(curves.shape[1])
    min_vals = curves.min(axis=0)
    max_vals = curves.max(axis=0)

    log_curves = np.log(np.maximum(curves, 1e-300))
    log_mean = log_curves.mean(axis=0)
    log_std = log_curves.std(axis=0, ddof=1) if n > 1 else np.zeros(curves.shape[1])

    sem = np.sqrt(var / n)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    histogram = _compute_histogram(curves)

    return ConfigStats(
        steps=baseline_steps,
        baseline_loss=baseline_loss,
        baseline_var=baseline_var,
        baseline_n=baseline_n,
        baseline_min=baseline_min,
        baseline_max=baseline_max,
        baseline_hist_counts=baseline_hist_counts,
        baseline_hist_edges=baseline_hist_edges,
        sgd_mean=mean,
        sgd_lower=mean - t_val * sem,
        sgd_upper=mean + t_val * sem,
        sgd_min=min_vals,
        sgd_max=max_vals,
        sgd_var=var,
        sgd_log_mean=log_mean,
        sgd_log_std=log_std,
        hist_counts=histogram,
        hist_edges=HIST_EDGES,
        n_runs=n,
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

    bl_key_cols = ["model.hidden_dim", "model.gamma", "data.noise_std", "model.model_seed"]
    data_cols = ["step", "test_loss"]

    # Phase 1: Baselines — small enough to load fully
    baseline_lf = _scan_sweep(baseline_dir)
    print(f"Loading baselines from {baseline_dir}...")
    baseline_df = baseline_lf.select(bl_key_cols + data_cols).collect()
    baseline_groups = baseline_df.partition_by(bl_key_cols, as_dict=True)

    print(f"Computing baselines ({len(baseline_groups)} configs)...")
    baselines = {}
    for key, group_df in baseline_groups.items():
        result = _compute_baseline_stats(group_df)
        if result is not None:
            baselines[key] = result
    del baseline_df, baseline_groups
    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — load per baseline key to avoid 7+ GB in memory
    sgd_lf = _scan_sweep(sgd_dir)
    bl_keys = sorted(baselines.keys())
    n_bl = len(bl_keys)
    print(f"Computing SGD statistics ({n_bl} baseline groups)...")
    stats = {}
    for i, bl_key in enumerate(bl_keys):
        # Load all batch sizes for this (width, gamma, noise, seed) at once
        chunk = (
            sgd_lf.filter(_build_filter(bl_key_cols, bl_key))
            .select(["training.batch_size"] + data_cols)
            .collect()
        )
        # Split locally by batch_size
        for bs_group in chunk.partition_by("training.batch_size"):
            batch_size = bs_group["training.batch_size"][0]
            key = bl_key + (batch_size,)
            result = _compute_sgd_config_stats(bs_group, *baselines[bl_key])
            if result is not None:
                stats[key] = result
        print(
            f"\r  SGD stats: {i + 1}/{n_bl} ({100 * (i + 1) / n_bl:.0f}%)",
            end="",
            flush=True,
        )

    print(f"\n  Computed {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def save_cache(stats: dict[tuple, ConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    for key, cs in stats.items():
        cache_data[key] = {
            "steps": cs.steps,
            "baseline_loss": cs.baseline_loss,
            "baseline_var": cs.baseline_var,
            "baseline_n": cs.baseline_n,
            "baseline_min": cs.baseline_min,
            "baseline_max": cs.baseline_max,
            "baseline_hist_counts": cs.baseline_hist_counts,
            "baseline_hist_edges": cs.baseline_hist_edges,
            "sgd_mean": cs.sgd_mean,
            "sgd_lower": cs.sgd_lower,
            "sgd_upper": cs.sgd_upper,
            "sgd_min": cs.sgd_min,
            "sgd_max": cs.sgd_max,
            "sgd_var": cs.sgd_var,
            "sgd_log_mean": cs.sgd_log_mean,
            "sgd_log_std": cs.sgd_log_std,
            "hist_counts": cs.hist_counts,
            "hist_edges": cs.hist_edges,
            "n_runs": cs.n_runs,
        }
    with open(path, "wb") as f:
        pickle.dump(cache_data, f)
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
    exp_config: dict, force_recompute: bool = False
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


def _percentiles_from_histogram(
    hist_counts: np.ndarray, hist_edges: np.ndarray, percentiles: list[float]
) -> np.ndarray:
    """Extract loss percentile curves from per-step histogram.

    Args:
        hist_counts: (n_steps, n_bins) bin counts at each step.
        hist_edges: (n_bins + 1,) bin edges in log10 space.
        percentiles: values in [0, 100].

    Returns:
        (len(percentiles), n_steps) array of loss values.
    """
    n_steps, n_bins = hist_counts.shape
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    cdf = np.cumsum(hist_counts, axis=1).astype(np.float64)
    cdf /= cdf[:, -1:]

    result = np.empty((len(percentiles), n_steps))
    for i, p in enumerate(percentiles):
        q = p / 100.0
        idx = (cdf < q).sum(axis=1)

        at_floor = idx == 0
        at_ceil = idx >= n_bins

        idx_safe = np.clip(idx, 1, n_bins - 1)
        step_range = np.arange(n_steps)
        cdf_lo = cdf[step_range, idx_safe - 1]
        cdf_hi = cdf[step_range, idx_safe]
        frac = (q - cdf_lo) / np.maximum(cdf_hi - cdf_lo, 1e-10)
        result[i] = bin_centers[idx_safe - 1] + frac * (
            bin_centers[idx_safe] - bin_centers[idx_safe - 1]
        )

        result[i, at_floor] = bin_centers[0]
        result[i, at_ceil] = bin_centers[-1]

    return 10.0**result


def save_fig(fig: plt.Figure, name: str) -> None:
    figures_path = _exp["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_path / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _welch_t_crit(se_a: np.ndarray, n_a: int, se_b: np.ndarray, n_b: int) -> np.ndarray:
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


def _significance_mask(s: ConfigStats) -> np.ndarray:
    """Compute where E[baseline] < E[SGD] at 95% confidence.

    For deterministic baselines (offline), this reduces to the one-sample
    test against the SGD CI. For stochastic baselines (online), uses
    Welch's two-sample t-test on the difference.
    """
    if s.baseline_var is None:
        return s.baseline_loss < s.sgd_lower

    # Welch's two-sample test
    diff = s.sgd_mean - s.baseline_loss
    se_bl = s.baseline_var / s.baseline_n
    se_sgd = s.sgd_var / s.n_runs
    se_diff = np.sqrt(se_bl + se_sgd)

    t_crit = _welch_t_crit(se_bl, s.baseline_n, se_sgd, s.n_runs)

    # Lower bound of 95% CI for (E[SGD] - E[baseline]) > 0
    return diff > t_crit * se_diff


def _ratio_ci(s: ConfigStats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SGD/baseline ratio with 95% CI.

    For deterministic baselines: divides SGD CI bounds by the constant.
    For stochastic baselines: delta method for the ratio of two means.
    """
    ratio = s.sgd_mean / s.baseline_loss

    if s.baseline_var is None:
        # Constant denominator: CI transforms linearly
        return ratio, s.sgd_lower / s.baseline_loss, s.sgd_upper / s.baseline_loss

    # Delta method: Var(Y/X) ≈ (Y/X)² × (Var(Y)/(n_Y·Y²) + Var(X)/(n_X·X²))
    safe_sgd = np.maximum(s.sgd_mean, 1e-30)
    safe_bl = np.maximum(s.baseline_loss, 1e-30)
    rel_var = s.sgd_var / (s.n_runs * safe_sgd**2) + s.baseline_var / (
        s.baseline_n * safe_bl**2
    )
    se_ratio = ratio * np.sqrt(rel_var)

    se_bl = s.baseline_var / s.baseline_n
    se_sgd = s.sgd_var / s.n_runs
    t_crit = _welch_t_crit(se_bl, s.baseline_n, se_sgd, s.n_runs)

    return ratio, ratio - t_crit * se_ratio, ratio + t_crit * se_ratio


def _suptitle_params(
    gamma: float, noise: float, n_seeds,
) -> str:
    """Build pipe-separated shared parameter string for figure suptitles."""
    baseline_bs = _exp["baseline_batch_size"]

    if baseline_bs is not None:
        regime = "Online (infinite data)"
        seeds_label = f"{_fmt_seeds(n_seeds)} batch seeds"
    else:
        regime = "Fixed train set (N=500)"
        seeds_label = f"{_fmt_seeds(n_seeds)} batch partitions"

    parts = [
        regime,
        f"{GAMMA_NAMES[gamma]} initialisation (\u03b3 = {gamma})",
        f"Label noise std = {noise}",
        seeds_label,
    ]
    return " | ".join(parts)


def _get_n_seeds(stats: dict[tuple, ConfigStats]):
    """Look up the number of batch seeds from any config."""
    sample_key = next(iter(stats), None)
    return stats[sample_key].n_runs if sample_key else "?"


def plot_loss_ratio(
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
    bl_label = _exp["baseline_label"]
    baseline_bs = _exp["baseline_batch_size"]

    # Width-independent labels
    if baseline_bs is not None:
        ratio_ylabel = f"$L_{{B={batch_size}}} \\,/\\, L_{{B={baseline_bs}}}$"
        bl_legend = f"B={baseline_bs}"
        sgd_legend = f"B={batch_size}"
        sig_label = f"$L_{{B={baseline_bs}}} < L_{{B={batch_size}}}$ (p<0.05)"
    else:
        ratio_ylabel = f"$L_{{\\mathrm{{SGD}}}} \\,/\\, L_{{\\mathrm{{{bl_label}}}}}$"
        bl_legend = bl_label
        sgd_legend = f"SGD (B={batch_size})"
        sig_label = f"$L_{{\\mathrm{{{bl_label}}}}} < L_{{\\mathrm{{SGD}}}}$ (p<0.05)"

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
            # Log-transformed CI of the arithmetic mean (delta method on log(X̄))
            # Var(log X̄) ≈ SEM²/μ², so CI = mean ×/÷ exp(t × SEM/mean)
            t_val = scipy_stats.t.ppf(0.975, df=max(s.n_runs - 1, 1))
            sem = np.sqrt(s.sgd_var / s.n_runs)
            relative_sem = sem / np.maximum(s.sgd_mean, 1e-30)
            ci_factor = np.exp(t_val * relative_sem)
            ax_loss.fill_between(
                s.steps,
                s.sgd_mean / ci_factor, s.sgd_mean * ci_factor,
                alpha=0.3, color="C1",
            )
            sig_95 = _significance_mask(s)
            ax_loss.fill_between(
                s.steps, 0, 1,
                where=sig_95, alpha=0.4, color="darkgreen",
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
            ratio_mean, ratio_lower, ratio_upper = _ratio_ci(s)
            ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
            ax_ratio.plot(
                s.steps, ratio_mean,
                label="Ratio \u00b1 95% CI", color="C1", linewidth=1.5,
            )
            ax_ratio.fill_between(
                s.steps, ratio_lower, ratio_upper, alpha=0.3, color="C1",
            )
            if col == 0:
                ax_ratio.set_ylabel(f"Width {width}\n{ratio_ylabel}")
            if is_bottom:
                ax_ratio.set_xlabel("Training step")
            ax_ratio.legend(loc="upper right", fontsize=7)

    n_batch_seeds = _get_n_seeds(stats)
    baseline_bs = _exp["baseline_batch_size"]
    if baseline_bs is not None:
        prefix = (
            f"Loss Ratio | batch size {baseline_bs} vs "
            f"batch size {batch_size}"
        )
    else:
        prefix = (
            f"Loss Ratio | GD loss vs expected SGD loss, "
            f"batch size = {batch_size}"
        )
    params = _suptitle_params(gamma, noise, n_batch_seeds)
    fig.suptitle(
        f"{prefix} | {params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_loss_variability(
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
    bl_label = _exp["baseline_label"]
    baseline_bs = _exp["baseline_batch_size"]

    # Width-independent labels
    if baseline_bs is not None:
        bl_legend = f"B={baseline_bs}"
        sgd_legend = f"B={batch_size}"
        cv_legend = f"B={batch_size}"
    else:
        bl_legend = bl_label
        sgd_legend = f"SGD (B={batch_size})"
        cv_legend = f"SGD (B={batch_size})"

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
                bl_p5, bl_p95 = _percentiles_from_histogram(
                    s.baseline_hist_counts, s.baseline_hist_edges, [5, 95],
                )
                ax_loss.fill_between(
                    s.steps, s.baseline_min, s.baseline_max,
                    alpha=0.25, color="C0", label="min\u2013max",
                )
                ax_loss.fill_between(
                    s.steps, bl_p5, bl_p95,
                    alpha=0.3, color="C0", label="P5\u2013P95",
                )
            ax_loss.plot(
                s.steps, s.sgd_mean,
                label=sgd_legend, color="C1", linewidth=1.5,
            )
            p5, p95 = _percentiles_from_histogram(
                s.hist_counts, s.hist_edges, [5, 95],
            )
            ax_loss.fill_between(
                s.steps, s.sgd_min, s.sgd_max,
                alpha=0.25, color="C1", label="min\u2013max",
            )
            ax_loss.fill_between(
                s.steps, p5, p95,
                alpha=0.3, color="C1", label="P5\u2013P95",
            )
            ax_loss.set_yscale("log")
            if col == 0:
                ax_loss.set_ylabel(f"Width {width}\nTest loss")
            if is_bottom:
                ax_loss.set_xlabel("Training step")
            ax_loss.legend(loc="upper right", fontsize=6)

    n_batch_seeds = _get_n_seeds(stats)
    baseline_bs = _exp["baseline_batch_size"]
    if baseline_bs is not None:
        prefix = (
            f"Loss Spread | batch size {baseline_bs} and "
            f"batch size {batch_size}"
        )
    else:
        prefix = f"SGD loss spread, batch size = {batch_size}"
    params = _suptitle_params(gamma, noise, n_batch_seeds)
    fig.suptitle(
        f"{prefix} | {params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def generate_all_plots(stats: dict[tuple, ConfigStats]) -> None:
    figures_path = _exp["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)

    # Derive parameter values from stats keys: (width, gamma, noise, model_seed, batch_size)
    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    total = len(gammas) * len(noise_levels) * len(batch_sizes)
    completed = 0

    print(f"Generating {total * 2} figures ({total} configs \u00d7 2 plot types)...")

    for gamma in gammas:
        for noise in noise_levels:
            for batch_size in batch_sizes:
                name = f"g{gamma}_noise{noise}_b{batch_size}"

                fig = plot_loss_ratio(
                    stats, gamma, noise, batch_size, widths, model_seeds,
                )
                save_fig(fig, f"ratio_{name}")

                fig = plot_loss_variability(
                    stats, gamma, noise, batch_size, widths, model_seeds,
                )
                save_fig(fig, f"variability_{name}")

                completed += 1
                print(
                    f"\r  Progress: {completed}/{total}"
                    f" ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

    print(f"\nAll plots saved to {figures_path}/")


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
    args = parser.parse_args()

    global _exp
    _exp = EXPERIMENTS[args.experiment]

    print(f"Experiment: {args.experiment}")
    print(f"Data: {_exp['base_path']}")

    stats = get_stats(_exp, force_recompute=args.recompute)
    generate_all_plots(stats)

    print("\nDone!")


if __name__ == "__main__":
    main()