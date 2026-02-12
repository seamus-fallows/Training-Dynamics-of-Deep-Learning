"""
GPH Sweep Analysis

Analyzes results from GPH experiment sweeps (offline or online) with:
- Parallel data loading across CPU cores
- Caching of computed statistics for fast plot iteration
- Two plot types: loss ratio with significance testing, loss variability with spread bands

Usage (run from the project root):

    python analysis/gph_analysis_loss_only.py offline
    python analysis/gph_analysis_loss_only.py online

Options:
    --recompute    Force recompute statistics (ignore cache)
    --workers N    Number of parallel workers (default: all CPUs)

Expects sweep outputs in outputs/gph_offline/ or outputs/gph_online/.
Saves figures to figures/gph_offline/ or figures/gph_online/.
Caches computed statistics in cache/gph_offline.pkl or cache/gph_online.pkl.
"""

import argparse
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# =============================================================================
# Configuration
# =============================================================================

WIDTHS = [10, 50, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]
MODEL_SEEDS = [0, 1]
BATCH_SIZES = [1, 2, 5, 10, 50]

GAMMA_MAX_STEPS = {0.75: 5000, 1.0: 8000, 1.5: 26000}
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

N_WORKERS = os.cpu_count()
MAX_BATCH_SEED = 10000

# Histogram for density bands: log10-spaced bins covering plausible loss range
LOG10_LOSS_RANGE = (-20, 5)
N_HIST_BINS = 334

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph_offline"),
        "cache_path": Path("cache/gph_offline.pkl"),
        "figures_path": Path("figures/gph_offline"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/gph_online"),
        "cache_path": Path("cache/gph_online.pkl"),
        "figures_path": Path("figures/gph_online"),
        "baseline_subdir": "large_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
        "regime_label": "Online (infinite data)",
    },
}

# Set in main() and worker_init()
_exp = None


# =============================================================================
# Data Loading
# =============================================================================


def _overrides_to_hash(overrides: dict) -> str:
    """Deterministic 12-char hex hash from override values (matches sweep.py)."""
    key = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _make_overrides(**kwargs) -> dict:
    """Build overrides dict with sweep.py key names."""
    key_map = {
        "gamma": "model.gamma",
        "max_steps": "max_steps",
        "model_seed": "model.model_seed",
        "noise": "data.noise_std",
        "hidden_dim": "model.hidden_dim",
        "batch_seed": "training.batch_seed",
        "batch_size": "training.batch_size",
    }
    return {key_map[k]: v for k, v in kwargs.items()}


def _run_dir(subdir: str, overrides: dict) -> Path:
    return _exp["base_path"] / subdir / _overrides_to_hash(overrides)


def _load_test_loss(path: Path) -> np.ndarray | None:
    """Load only test_loss from history.npz."""
    history_file = path / "history.npz"
    if not history_file.exists():
        return None
    try:
        with np.load(history_file) as data:
            return data["test_loss"]
    except Exception:
        return None


def _load_steps_and_loss(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load step and test_loss from history.npz."""
    history_file = path / "history.npz"
    if not history_file.exists():
        return None
    try:
        with np.load(history_file) as data:
            return data["step"], data["test_loss"]
    except Exception:
        return None


def _load_baseline(
    width: int, gamma: float, noise: float, model_seed: int
) -> tuple | None:
    """Load baseline loss curve.

    Offline (full_batch): single deterministic GD run.
        Returns (steps, loss, None, 1, None, None, None, None).
    Online (large_batch): Welford mean+variance+min+max+histogram over batch seeds.
        Returns (steps, mean, var, n, min, max, hist_counts, hist_edges).
    """
    max_steps = GAMMA_MAX_STEPS[gamma]
    subdir = _exp["baseline_subdir"]

    if subdir == "full_batch":
        overrides = _make_overrides(
            gamma=gamma,
            max_steps=max_steps,
            model_seed=model_seed,
            noise=noise,
            hidden_dim=width,
        )
        result = _load_steps_and_loss(_run_dir(subdir, overrides))
        if result is None:
            return None
        steps, loss = result
        return steps, loss, None, 1, None, None, None, None
    else:
        n = 0
        mean = None
        m2 = None
        min_vals = None
        max_vals = None
        hist_edges = np.linspace(*LOG10_LOSS_RANGE, N_HIST_BINS + 1)
        histogram = None
        steps = None
        for batch_seed in range(MAX_BATCH_SEED):
            overrides = _make_overrides(
                gamma=gamma,
                max_steps=max_steps,
                model_seed=model_seed,
                noise=noise,
                hidden_dim=width,
                batch_seed=batch_seed,
            )
            result = _load_steps_and_loss(_run_dir(subdir, overrides))
            if result is None:
                continue
            s, loss = result
            n += 1
            if mean is None:
                steps = s
                mean = loss.astype(np.float64)
                m2 = np.zeros_like(mean)
                min_vals = loss.copy()
                max_vals = loss.copy()
                histogram = np.zeros((len(loss), N_HIST_BINS), dtype=np.int32)
            else:
                delta = loss - mean
                mean += delta / n
                m2 += delta * (loss - mean)
                np.minimum(min_vals, loss, out=min_vals)
                np.maximum(max_vals, loss, out=max_vals)

            # Histogram accumulation
            log10_loss = np.log10(np.maximum(loss, 10.0 ** LOG10_LOSS_RANGE[0]))
            bin_idx = np.digitize(log10_loss, hist_edges) - 1
            np.clip(bin_idx, 0, N_HIST_BINS - 1, out=bin_idx)
            histogram[np.arange(len(loss)), bin_idx] += 1

        if n == 0 or steps is None:
            return None
        var = m2 / (n - 1) if n > 1 else np.zeros_like(mean)
        return steps, mean, var, n, min_vals, max_vals, histogram, hist_edges


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
    sgd_log_mean: np.ndarray  # Welford mean of log(loss)
    sgd_log_std: np.ndarray  # std of log(loss), for log-space spread bands
    hist_counts: np.ndarray  # (n_steps, N_HIST_BINS) loss histogram in log10 space
    hist_edges: np.ndarray  # (N_HIST_BINS + 1,) bin edges in log10 space
    n_runs: int


def _compute_sgd_stats(
    width: int,
    gamma: float,
    noise: float,
    model_seed: int,
    batch_size: int,
    baseline_steps: np.ndarray,
    baseline_loss: np.ndarray,
    baseline_var: np.ndarray | None,
    baseline_n: int,
    baseline_min: np.ndarray | None,
    baseline_max: np.ndarray | None,
    baseline_hist_counts: np.ndarray | None,
    baseline_hist_edges: np.ndarray | None,
) -> ConfigStats | None:
    """Compute SGD statistics using Welford's online algorithm.

    Uses O(steps) memory instead of O(n_runs * steps) by computing
    mean, variance, min, max incrementally.
    """
    max_steps = GAMMA_MAX_STEPS[gamma]
    subdir = _exp["sgd_subdir"]

    n = 0
    mean = None
    m2 = None
    log_mean = None
    log_m2 = None
    min_vals = None
    max_vals = None
    hist_edges = np.linspace(*LOG10_LOSS_RANGE, N_HIST_BINS + 1)
    histogram = None

    for batch_seed in range(MAX_BATCH_SEED):
        overrides = _make_overrides(
            gamma=gamma,
            max_steps=max_steps,
            model_seed=model_seed,
            noise=noise,
            hidden_dim=width,
            batch_seed=batch_seed,
            batch_size=batch_size,
        )
        loss = _load_test_loss(_run_dir(subdir, overrides))
        if loss is None:
            continue

        n += 1
        log_loss = np.log(np.maximum(loss, 1e-300))
        if mean is None:
            mean = loss.astype(np.float64)
            m2 = np.zeros_like(mean)
            log_mean = log_loss.astype(np.float64)
            log_m2 = np.zeros_like(log_mean)
            min_vals = loss.copy()
            max_vals = loss.copy()
            histogram = np.zeros((len(loss), N_HIST_BINS), dtype=np.int32)
        else:
            delta = loss - mean
            mean += delta / n
            m2 += delta * (loss - mean)
            log_delta = log_loss - log_mean
            log_mean += log_delta / n
            log_m2 += log_delta * (log_loss - log_mean)
            np.minimum(min_vals, loss, out=min_vals)
            np.maximum(max_vals, loss, out=max_vals)

        # Histogram accumulation
        log10_loss = np.log10(np.maximum(loss, 10.0 ** LOG10_LOSS_RANGE[0]))
        bin_idx = np.digitize(log10_loss, hist_edges) - 1
        np.clip(bin_idx, 0, N_HIST_BINS - 1, out=bin_idx)
        histogram[np.arange(len(loss)), bin_idx] += 1

    if n == 0:
        return None

    var = m2 / (n - 1) if n > 1 else np.zeros_like(mean)
    log_var = log_m2 / (n - 1) if n > 1 else np.zeros_like(log_mean)
    log_std = np.sqrt(log_var)
    sem = np.sqrt(var / n)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0

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
        hist_edges=hist_edges,
        n_runs=n,
    )


# =============================================================================
# Parallel Computation
# =============================================================================


def _worker_init(exp_config):
    global _exp
    _exp = exp_config


def _baseline_wrapper(args):
    width, gamma, noise, model_seed = args
    return args, _load_baseline(width, gamma, noise, model_seed)


def _sgd_wrapper(args):
    (
        config_key,
        width,
        gamma,
        noise,
        model_seed,
        batch_size,
        bl_steps,
        bl_loss,
        bl_var,
        bl_n,
        bl_min,
        bl_max,
        bl_hist_counts,
        bl_hist_edges,
    ) = args
    result = _compute_sgd_stats(
        width, gamma, noise, model_seed, batch_size,
        bl_steps, bl_loss, bl_var, bl_n,
        bl_min, bl_max, bl_hist_counts, bl_hist_edges,
    )
    return config_key, result


def compute_all_stats(
    exp_config: dict, n_workers: int = N_WORKERS
) -> dict[tuple, ConfigStats]:
    global _exp
    _exp = exp_config

    # Phase 1: Compute baselines
    baseline_configs = [
        (width, gamma, noise, model_seed)
        for gamma in GAMMAS
        for noise in NOISE_LEVELS
        for width in WIDTHS
        for model_seed in MODEL_SEEDS
    ]

    print(f"Loading baselines ({len(baseline_configs)} configs)...")
    baselines = {}

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init, initargs=(exp_config,)
    ) as executor:
        futures = {
            executor.submit(_baseline_wrapper, cfg): cfg for cfg in baseline_configs
        }
        done = 0
        for future in as_completed(futures):
            cfg, result = future.result()
            if result is not None:
                baselines[cfg] = result
            done += 1
            print(
                f"\r  Baselines: {done}/{len(baseline_configs)}",
                end="",
                flush=True,
            )

    print(f"\n  Loaded {len(baselines)}/{len(baseline_configs)} baselines")

    # Phase 2: Compute SGD stats
    sgd_args = []
    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for batch_size in BATCH_SIZES:
                for width in WIDTHS:
                    for model_seed in MODEL_SEEDS:
                        bl_key = (width, gamma, noise, model_seed)
                        if bl_key not in baselines:
                            continue
                        (bl_steps, bl_loss, bl_var, bl_n,
                         bl_min, bl_max, bl_hist_counts, bl_hist_edges) = baselines[bl_key]
                        config_key = (width, gamma, noise, model_seed, batch_size)
                        sgd_args.append(
                            (
                                config_key,
                                width,
                                gamma,
                                noise,
                                model_seed,
                                batch_size,
                                bl_steps,
                                bl_loss,
                                bl_var,
                                bl_n,
                                bl_min,
                                bl_max,
                                bl_hist_counts,
                                bl_hist_edges,
                            )
                        )

    print(f"Computing SGD statistics ({len(sgd_args)} configs)...")
    stats = {}

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init, initargs=(exp_config,)
    ) as executor:
        futures = {executor.submit(_sgd_wrapper, cfg): cfg for cfg in sgd_args}
        done = 0
        for future in as_completed(futures):
            config_key, result = future.result()
            if result is not None:
                stats[config_key] = result
            done += 1
            print(
                f"\r  SGD stats: {done}/{len(sgd_args)} ({100 * done / len(sgd_args):.0f}%)",
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
    exp_config: dict, force_recompute: bool = False, n_workers: int = N_WORKERS
) -> dict[tuple, ConfigStats]:
    cache_path = exp_config["cache_path"]
    if not force_recompute:
        stats = load_cache(cache_path)
        if stats is not None:
            print(f"Loaded {len(stats)} configurations from cache")
            return stats

    stats = compute_all_stats(exp_config, n_workers=n_workers)
    save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting
# =============================================================================


def _fmt_seeds(n) -> str:
    """Format seed count for titles, e.g. 10000 → '$10^4$'."""
    if isinstance(n, int) and n >= 1000:
        exp = int(np.log10(n))
        if 10**exp == n:
            return f"$10^{exp}$"
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
        idx = np.array([np.searchsorted(cdf[t], q) for t in range(n_steps)])

        at_floor = idx == 0
        at_ceil = idx >= n_bins

        idx_safe = np.clip(idx, 1, n_bins - 1)
        cdf_lo = cdf[np.arange(n_steps), idx_safe - 1]
        cdf_hi = cdf[np.arange(n_steps), idx_safe]
        frac = (q - cdf_lo) / np.maximum(cdf_hi - cdf_lo, 1e-10)
        result[i] = bin_centers[idx_safe - 1] + frac * (bin_centers[idx_safe] - bin_centers[idx_safe - 1])

        result[i, at_floor] = bin_centers[0]
        result[i, at_ceil] = bin_centers[-1]

    return 10.0 ** result


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


def plot_combined(
    stats: dict[tuple, ConfigStats],
    gamma: float, noise: float, batch_size: int, model_seed: int,
) -> plt.Figure:
    """Combined loss ratio + variability analysis for a single model seed.

    Layout uses two subfigures (Loss Ratio / Loss Variability), each with
    2 rows × len(WIDTHS) columns.
    """
    bl_label = _exp["baseline_label"]
    regime = _exp["regime_label"]
    baseline_bs = _exp["baseline_batch_size"]

    # Build labels once (same for all widths within a figure)
    if baseline_bs is not None:
        ratio_ylabel = f"$L_{{B={batch_size}}} \\,/\\, L_{{B={baseline_bs}}}$"
        bl_legend = f"B={baseline_bs}"
        sgd_legend = f"B={batch_size}"
        sig_label = (
            f"$L_{{B={baseline_bs}}} < L_{{B={batch_size}}}$ (p<0.05)"
        )
        cv_legend = f"B={batch_size}"
    else:
        ratio_ylabel = (
            f"$L_{{\\mathrm{{SGD}}}} \\,/\\, L_{{\\mathrm{{{bl_label}}}}}$"
        )
        bl_legend = bl_label
        sgd_legend = "SGD"
        sig_label = (
            f"$L_{{\\mathrm{{{bl_label}}}}} < L_{{\\mathrm{{SGD}}}}$"
            " (p<0.05)"
        )
        cv_legend = "SGD"

    n_cols = len(WIDTHS)
    fig = plt.figure(figsize=(5 * n_cols, 15))
    subfigs = fig.subfigures(2, 1, hspace=0.07)

    subfigs[0].suptitle("Loss Ratio", fontsize=11, fontweight="bold")
    axes_top = subfigs[0].subplots(2, n_cols, squeeze=False)

    subfigs[1].suptitle("Loss Variability", fontsize=11, fontweight="bold")
    axes_bot = subfigs[1].subplots(2, n_cols, squeeze=False)

    for col, width in enumerate(WIDTHS):
        ax_ratio = axes_top[0, col]
        ax_loss_ratio = axes_top[1, col]
        ax_cv = axes_bot[0, col]
        ax_loss_var = axes_bot[1, col]

        key = (width, gamma, noise, model_seed, batch_size)

        if key not in stats:
            ax_ratio.set_title(f"width={width} (no data)")
            for ax in (ax_ratio, ax_loss_ratio, ax_cv, ax_loss_var):
                ax.set_xlabel("Training step")
            ax_ratio.set_ylabel(ratio_ylabel)
            ax_loss_ratio.set_ylabel("Test loss")
            ax_cv.set_ylabel(r"$\sigma(\log L)$")
            ax_loss_var.set_ylabel("Test loss")
            continue

        s = stats[key]

        # === Row 0: Loss ratio ===
        ratio_mean, ratio_lower, ratio_upper = _ratio_ci(s)
        ax_ratio.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax_ratio.plot(
            s.steps, ratio_mean, label="Ratio ± 95% CI",
            color="C1", linewidth=1.5,
        )
        ax_ratio.fill_between(
            s.steps, ratio_lower, ratio_upper, alpha=0.3, color="C1",
        )
        ax_ratio.set_xlabel("Training step")
        ax_ratio.set_ylabel(ratio_ylabel)
        ax_ratio.set_title(f"width={width}")
        ax_ratio.legend(loc="upper right", fontsize=7)

        # === Row 1: Loss with CI + significance shading ===
        ax_loss_ratio.plot(
            s.steps, s.baseline_loss, label=bl_legend,
            color="C0", linewidth=1.5,
        )
        ax_loss_ratio.plot(
            s.steps, s.sgd_mean, label=sgd_legend,
            color="C1", linewidth=1.5,
        )
        # Log-transformed CI of the arithmetic mean (delta method on log(X̄))
        # Var(log X̄) ≈ SEM²/μ², so CI = mean ×/÷ exp(t × SEM/mean)
        t_val = scipy_stats.t.ppf(0.975, df=max(s.n_runs - 1, 1))
        sem = np.sqrt(s.sgd_var / s.n_runs)
        relative_sem = sem / np.maximum(s.sgd_mean, 1e-30)
        ci_factor = np.exp(t_val * relative_sem)
        ax_loss_ratio.fill_between(
            s.steps,
            s.sgd_mean / ci_factor,
            s.sgd_mean * ci_factor,
            alpha=0.3, color="C1",
        )
        sig_95 = _significance_mask(s)
        ax_loss_ratio.fill_between(
            s.steps, 0, 1, where=sig_95, alpha=0.4, color="darkgreen",
            transform=ax_loss_ratio.get_xaxis_transform(), label=sig_label,
        )
        ax_loss_ratio.set_yscale("log")
        ax_loss_ratio.set_xlabel("Training step")
        ax_loss_ratio.set_ylabel("Test loss")
        ax_loss_ratio.legend(loc="upper right", fontsize=6)

        # === Row 2: σ(log L) ===
        ax_cv.plot(
            s.steps, s.sgd_log_std, color="C1",
            linewidth=1.5, label=cv_legend,
        )
        ax_cv.set_xlabel("Training step")
        ax_cv.set_ylabel(r"$\sigma(\log L)$")
        ax_cv.set_title(f"width={width}")
        ax_cv.legend(loc="upper right", fontsize=7)

        # === Row 3: Loss with spread bands ===
        ax_loss_var.plot(
            s.steps, s.baseline_loss, label=bl_legend,
            color="C0", linewidth=1.5,
        )
        # Baseline bands (online experiment only)
        if s.baseline_min is not None:
            bl_p5, bl_p95 = _percentiles_from_histogram(
                s.baseline_hist_counts, s.baseline_hist_edges, [5, 95],
            )
            ax_loss_var.fill_between(
                s.steps, s.baseline_min, s.baseline_max,
                alpha=0.25, color="C0", label="min–max",
            )
            ax_loss_var.fill_between(
                s.steps, bl_p5, bl_p95,
                alpha=0.3, color="C0", label="P5–P95",
            )
        ax_loss_var.plot(
            s.steps, s.sgd_mean, label=sgd_legend,
            color="C1", linewidth=1.5,
        )
        p5, p95 = _percentiles_from_histogram(
            s.hist_counts, s.hist_edges, [5, 95],
        )
        ax_loss_var.fill_between(
            s.steps, s.sgd_min, s.sgd_max,
            alpha=0.25, color="C1", label="min–max",
        )
        ax_loss_var.fill_between(
            s.steps, p5, p95,
            alpha=0.3, color="C1", label="P5–P95",
        )
        ax_loss_var.set_yscale("log")
        ax_loss_var.set_xlabel("Training step")
        ax_loss_var.set_ylabel("Test loss")
        ax_loss_var.legend(loc="upper right", fontsize=6)

    # Figure suptitle
    sample_key = next(
        (k for k in stats if k[3] == model_seed and k[4] == batch_size),
        None,
    )
    n_seeds = stats[sample_key].n_runs if sample_key else "?"

    if baseline_bs is not None:
        suptitle = (
            f"B={batch_size} vs B={baseline_bs} — {regime}, "
            f"model seed {model_seed}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise std={noise}, "
            f"averaged over {_fmt_seeds(n_seeds)} batch seeds"
        )
    else:
        suptitle = (
            f"SGD (B={batch_size}) vs GD — {regime}, "
            f"model seed {model_seed}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise std={noise}, "
            f"averaged over {_fmt_seeds(n_seeds)} batch seeds"
        )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


def generate_all_plots(stats: dict[tuple, ConfigStats]) -> None:
    figures_path = _exp["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)

    total = (
        len(GAMMAS) * len(NOISE_LEVELS) * len(BATCH_SIZES) * len(MODEL_SEEDS)
    )
    completed = 0

    print(f"Generating {total} plots...")

    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for batch_size in BATCH_SIZES:
                for model_seed in MODEL_SEEDS:
                    fig = plot_combined(
                        stats, gamma, noise, batch_size, model_seed,
                    )
                    save_fig(
                        fig,
                        f"analysis_g{gamma}_noise{noise}"
                        f"_b{batch_size}_seed{model_seed}",
                    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Number of parallel workers (default: {N_WORKERS})",
    )
    args = parser.parse_args()

    global _exp
    _exp = EXPERIMENTS[args.experiment]

    print(f"Experiment: {args.experiment}")
    print(f"Data: {_exp['base_path']}")

    stats = get_stats(_exp, force_recompute=args.recompute, n_workers=args.workers)
    generate_all_plots(stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
