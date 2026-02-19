"""
GPH Metrics Analysis

Analyzes results from GPH experiment sweeps (offline or online) that include
drift (grad_norm_squared) and diffusion (trace_gradient_covariance) alongside
test_loss.

Generates two figure types per (gamma, noise, width, batch_size) combo:
- Drift and Diffusion: loss, drift, diffusion, drift/diffusion ratio
- Metric Spread: σ(log metric) and min-max bands for drift and diffusion

Usage (run from the project root):

    python analysis/gph_analysis_metrics.py offline
    python analysis/gph_analysis_metrics.py online

Options:
    --recompute       Force recompute statistics (ignore cache)
    --sort-parquet    Sort parquet files by key columns for efficient predicate
                      pushdown (one-time preprocessing step), then exit

Expects sweep outputs in outputs/gph/gph_offline_metrics/ or outputs/gph/gph_online_metrics/.
Saves figures to figures/gph_offline_metrics/ or figures/gph_online_metrics/.
Caches computed statistics in .analysis_cache/gph_offline_metrics.pkl or .analysis_cache/gph_online_metrics.pkl.
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

METRIC_COLS = [
    "test_loss",
    "grad_norm_squared",
    "trace_gradient_covariance",
]

BL_KEY_COLS = ["model.hidden_dim", "model.gamma", "data.noise_std", "model.model_seed"]
BATCH_KEY_COLS = BL_KEY_COLS[:3]  # columns used for batched parquet reads

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph/gph_offline_metrics"),
        "cache_path": Path(".analysis_cache/gph_offline_metrics.pkl"),
        "figures_path": Path("figures/gph_offline_metrics"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/gph/gph_online_metrics"),
        "cache_path": Path(".analysis_cache/gph_online_metrics.pkl"),
        "figures_path": Path("figures/gph_online_metrics"),
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
class MetricStats:
    """Per-metric statistics with precomputed plotting quantities."""

    mean: np.ndarray       # (n_steps,)
    n: int
    min_vals: np.ndarray | None   # None for deterministic baseline
    max_vals: np.ndarray | None
    spread_lo: np.ndarray | None  # mean-centered 90% band
    spread_hi: np.ndarray | None
    log_std: np.ndarray | None    # std of log(metric), None for deterministic
    ci_lo: np.ndarray      # 95% CI lower (log-space delta method)
    ci_hi: np.ndarray      # 95% CI upper


@dataclass
class ConfigStats:
    """Statistics for a single (width, gamma, noise, model_seed, batch_size) configuration.

    baseline and sgd dicts are keyed by metric name. Keys include METRIC_COLS
    plus "drift_over_diffusion" (computed per-run from drift / diffusion).
    """

    steps: np.ndarray
    baseline: dict[str, MetricStats]
    sgd: dict[str, MetricStats]


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_metric_curves(df: pl.DataFrame, metric_name: str) -> np.ndarray:
    """Extract metric curves as a (n_runs, n_steps) numpy array."""
    return np.vstack(df[metric_name].to_list())


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


def _make_metric_stats(curves: np.ndarray) -> MetricStats:
    """Build a MetricStats from a (n_runs, n_steps) array of positive values."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1,
            min_vals=None, max_vals=None,
            spread_lo=None, spread_hi=None,
            log_std=None, ci_lo=mean, ci_hi=mean,
        )

    spread_lo, spread_hi = _mean_centered_spread(curves, mean)
    log_std = np.log(np.maximum(curves, 1e-300)).std(axis=0, ddof=1)

    # Precompute log-space 95% CI (delta method on log(X̄))
    var = curves.var(axis=0, ddof=1)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(var / n)
    relative_sem = sem / np.maximum(np.abs(mean), 1e-30)
    ci_factor = np.exp(np.minimum(t_val * relative_sem, 700))

    return MetricStats(
        mean=mean, n=n,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
        spread_lo=spread_lo,
        spread_hi=spread_hi,
        log_std=log_std,
        ci_lo=mean / ci_factor,
        ci_hi=mean * ci_factor,
    )


def _compute_all_metric_stats(subset: pl.DataFrame) -> dict[str, MetricStats]:
    """Compute stats for all metrics including drift/diffusion ratio."""
    all_curves = {}
    metric_stats = {}
    for col in METRIC_COLS:
        curves = _extract_metric_curves(subset, col)
        all_curves[col] = curves
        metric_stats[col] = _make_metric_stats(curves)

    drift = all_curves["grad_norm_squared"]
    diffusion = all_curves["trace_gradient_covariance"]
    ratio_curves = drift / np.maximum(diffusion, 1e-30)
    metric_stats["drift_over_diffusion"] = _make_metric_stats(ratio_curves)

    return metric_stats


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
        .select(BL_KEY_COLS + ["step"] + METRIC_COLS)
        .collect()
    )
    baseline_groups = baseline_df.partition_by(BL_KEY_COLS, as_dict=True)

    print(f"Computing baselines ({len(baseline_groups)} configs)...")
    baselines: dict[tuple, tuple[np.ndarray, dict[str, MetricStats]]] = {}
    for key, group_df in baseline_groups.items():
        if len(group_df) == 0:
            continue
        steps = np.array(group_df["step"][0])
        baselines[key] = (steps, _compute_all_metric_stats(group_df))
    del baseline_df, baseline_groups
    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — batch by (width, gamma, noise) to reduce parquet scans.
    # Each batch loads all model seeds and batch sizes for one (width, gamma, noise)
    # combination, keeping memory bounded while reducing I/O.
    sgd_lf = pl.scan_parquet(sgd_dir / "results.parquet")
    # "step" is omitted — identical across runs, already stored in baselines
    sgd_select = BL_KEY_COLS + ["training.batch_size"] + METRIC_COLS

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
            if len(group_df) == 0:
                continue
            bl_steps, bl_metrics = bl
            key = bl_key + (batch_size,)
            stats[key] = ConfigStats(
                steps=bl_steps,
                baseline=bl_metrics,
                sgd=_compute_all_metric_stats(group_df),
            )

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
    cache_data = {
        key: {
            "steps": cs.steps,
            "baseline": {m: vars(ms) for m, ms in cs.baseline.items()},
            "sgd": {m: vars(ms) for m, ms in cs.sgd.items()},
        }
        for key, cs in stats.items()
    }
    with open(path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        return {
            key: ConfigStats(
                steps=d["steps"],
                baseline={m: MetricStats(**ms) for m, ms in d["baseline"].items()},
                sgd={m: MetricStats(**ms) for m, ms in d["sgd"].items()},
            )
            for key, d in cache_data.items()
        }
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
    if sample_key is None:
        return "?"
    first_metric = next(iter(stats[sample_key].sgd.values()))
    return first_metric.n


# -- Drift and Diffusion figure ------------------------------------------------

ROW_INFO = [
    {"metric": "test_loss", "ylabel": "Test loss", "yscale": "log"},
    {"metric": "grad_norm_squared", "ylabel": r"Drift ($\|\nabla L\|^2$)", "yscale": "log"},
    {"metric": "trace_gradient_covariance", "ylabel": r"Diffusion (Tr($\Sigma$))", "yscale": "log"},
    {"metric": "drift_over_diffusion", "ylabel": "Drift / Diffusion", "yscale": "log"},
]


def plot_metrics(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    gamma: float,
    noise: float,
    width: int,
    batch_size: int,
    model_seeds: list,
) -> plt.Figure:
    """Metrics: 4 rows (loss, drift, diffusion, drift/diffusion) x model_seeds columns."""
    baseline_bs = exp_config["baseline_batch_size"]

    if baseline_bs is not None:
        bl_legend = f"B={baseline_bs}"
        sgd_legend = f"B={batch_size}"
    else:
        bl_legend = exp_config["baseline_label"]
        sgd_legend = f"SGD (B={batch_size})"

    n_cols = len(model_seeds)
    n_rows = len(ROW_INFO)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    for col, model_seed in enumerate(model_seeds):
        key = (width, gamma, noise, model_seed, batch_size)

        for row, info in enumerate(ROW_INFO):
            ax = axes[row, col]

            if key not in stats:
                if row == 0:
                    ax.set_title(f"Model Seed {model_seed}")
                if col == 0:
                    ax.set_ylabel(info["ylabel"])
                if row == n_rows - 1:
                    ax.set_xlabel("Training step")
                continue

            s = stats[key]
            bl_ms = s.baseline[info["metric"]]
            sgd_ms = s.sgd[info["metric"]]

            ax.plot(s.steps, bl_ms.mean, label=bl_legend, color="C0", linewidth=1.5)
            if bl_ms.n > 1:
                ax.fill_between(
                    s.steps, bl_ms.ci_lo, bl_ms.ci_hi,
                    alpha=0.3, color="C0", label=f"{bl_legend} 95% CI",
                )

            ax.plot(s.steps, sgd_ms.mean, label=sgd_legend, color="C1", linewidth=1.5)
            ax.fill_between(
                s.steps, sgd_ms.ci_lo, sgd_ms.ci_hi,
                alpha=0.3, color="C1", label=f"{sgd_legend} 95% CI",
            )

            ax.set_yscale(info["yscale"])
            if row == 0:
                ax.set_title(f"Model Seed {model_seed}")
            if col == 0:
                ax.set_ylabel(info["ylabel"])
            if row == n_rows - 1:
                ax.set_xlabel("Training step")
            ax.legend(loc="upper right", fontsize=6)

    n_batch_seeds = _get_n_seeds(stats)
    if baseline_bs is not None:
        prefix = (
            f"Drift and diffusion | batch size {baseline_bs} vs "
            f"{batch_size} over {_fmt_seeds(n_batch_seeds)} batch seeds "
            f"| Width {width}"
        )
    else:
        prefix = (
            f"Drift and diffusion | GD vs SGD over "
            f"{_fmt_seeds(n_batch_seeds)} batch partitions | batch size = {batch_size} "
            f"| Width {width}"
        )
    params = _suptitle_params(exp_config, gamma, noise)
    fig.suptitle(
        f"{prefix} | {params}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# -- Drift/Diffusion Spread figure ---------------------------------------------

SPREAD_METRICS = [
    {
        "metric": "grad_norm_squared",
        "sigma_ylabel": r"$\sigma(\log$ drift$)$",
        "bands_ylabel": r"Drift ($\|\nabla L\|^2$)",
    },
    {
        "metric": "trace_gradient_covariance",
        "sigma_ylabel": r"$\sigma(\log$ diffusion$)$",
        "bands_ylabel": r"Diffusion (Tr($\Sigma$))",
    },
]


def plot_metric_spread(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    gamma: float,
    noise: float,
    width: int,
    batch_size: int,
    model_seeds: list,
) -> plt.Figure:
    """Metric spread: 4 rows (sigma + bands for drift, sigma + bands for diffusion).

    For each of drift and diffusion:
      - sigma(log metric) row: SGD spread (and baseline if stochastic)
      - bands row: baseline + SGD mean with min-max and 90% around-mean bands
    """
    baseline_bs = exp_config["baseline_batch_size"]

    if baseline_bs is not None:
        bl_legend = f"B={baseline_bs}"
        sgd_legend = f"B={batch_size}"
    else:
        bl_legend = exp_config["baseline_label"]
        sgd_legend = f"SGD (B={batch_size})"

    n_cols = len(model_seeds)
    n_rows = 2 * len(SPREAD_METRICS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    for col, model_seed in enumerate(model_seeds):
        key = (width, gamma, noise, model_seed, batch_size)

        for m_idx, m_info in enumerate(SPREAD_METRICS):
            row_sigma = m_idx * 2
            row_bands = m_idx * 2 + 1
            ax_sigma = axes[row_sigma, col]
            ax_bands = axes[row_bands, col]

            if key not in stats:
                if m_idx == 0:
                    ax_sigma.set_title(f"Model Seed {model_seed}")
                if col == 0:
                    ax_sigma.set_ylabel(m_info["sigma_ylabel"])
                    ax_bands.set_ylabel(m_info["bands_ylabel"])
                if m_idx == len(SPREAD_METRICS) - 1:
                    ax_sigma.set_xlabel("Training step")
                    ax_bands.set_xlabel("Training step")
                continue

            s = stats[key]
            bl_ms = s.baseline[m_info["metric"]]
            sgd_ms = s.sgd[m_info["metric"]]

            # === Sigma row ===
            if sgd_ms.log_std is not None:
                ax_sigma.plot(
                    s.steps, sgd_ms.log_std,
                    color="C1", linewidth=1.5, label=sgd_legend,
                )
            if bl_ms.log_std is not None:
                ax_sigma.plot(
                    s.steps, bl_ms.log_std,
                    color="C0", linewidth=1.5, label=bl_legend,
                )
            if m_idx == 0:
                ax_sigma.set_title(f"Model Seed {model_seed}")
            if col == 0:
                ax_sigma.set_ylabel(m_info["sigma_ylabel"])
            ax_sigma.legend(loc="upper right", fontsize=7)

            # === Bands row ===
            ax_bands.plot(
                s.steps, bl_ms.mean,
                label=bl_legend, color="C0", linewidth=1.5,
            )
            if bl_ms.min_vals is not None:
                ax_bands.fill_between(
                    s.steps, bl_ms.min_vals, bl_ms.max_vals,
                    alpha=0.25, color="C0", label=f"{bl_legend} min\u2013max",
                )
                ax_bands.fill_between(
                    s.steps, bl_ms.spread_lo, bl_ms.spread_hi,
                    alpha=0.3, color="C0", label=f"{bl_legend} 90% around mean",
                )

            ax_bands.plot(
                s.steps, sgd_ms.mean,
                label=sgd_legend, color="C1", linewidth=1.5,
            )
            if sgd_ms.min_vals is not None:
                ax_bands.fill_between(
                    s.steps, sgd_ms.min_vals, sgd_ms.max_vals,
                    alpha=0.25, color="C1", label=f"{sgd_legend} min\u2013max",
                )
                ax_bands.fill_between(
                    s.steps, sgd_ms.spread_lo, sgd_ms.spread_hi,
                    alpha=0.3, color="C1", label=f"{sgd_legend} 90% around mean",
                )
            ax_bands.set_yscale("log")
            if col == 0:
                ax_bands.set_ylabel(m_info["bands_ylabel"])
            if m_idx == len(SPREAD_METRICS) - 1:
                ax_bands.set_xlabel("Training step")
            ax_bands.legend(loc="upper right", fontsize=6)

    n_batch_seeds = _get_n_seeds(stats)
    if baseline_bs is not None:
        prefix = (
            f"Drift and diffusion across "
            f"{_fmt_seeds(n_batch_seeds)} batch seeds | batch size {baseline_bs} and "
            f"{batch_size} | Width {width}"
        )
    else:
        prefix = (
            f"Drift and diffusion across "
            f"{_fmt_seeds(n_batch_seeds)} batch partitions | batch size = {batch_size} "
            f"| Width {width}"
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
    plot_type, gamma, noise, width, batch_size, model_seeds, filename = task
    stats = _worker_ctx["stats"]
    exp = _worker_ctx["exp"]

    plot_fn = plot_metrics if plot_type == "metrics" else plot_metric_spread
    fig = plot_fn(exp, stats, gamma, noise, width, batch_size, model_seeds)
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
            for width in widths:
                for batch_size in batch_sizes:
                    name = f"g{gamma}_noise{noise}_w{width}_b{batch_size}"
                    common = (gamma, noise, width, batch_size, model_seeds)
                    tasks.append(("metrics", *common, name))
                    tasks.append(("spread", *common, f"spread_{name}"))

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
    parser = argparse.ArgumentParser(description="Analyze GPH sweep metrics results")
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
