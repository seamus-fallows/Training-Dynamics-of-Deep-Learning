"""
GPH Metrics Analysis

Analyzes results from GPH experiment sweeps (offline or online) that include
drift (grad_norm_squared) and diffusion (trace_gradient_covariance) alongside
test_loss.

Generates one figure per (width, noise, model_seed, batch_size) combo:
  Row 1: Test loss (baseline + SGD mean)
  Row 2: Drift (‖∇L‖²)
  Row 3: Diffusion (Tr(Σ))
  Row 4: Drift / Diffusion ratio
  Columns: γ = 0.75 (NTK), γ = 1.0 (Mean-Field), γ = 1.5 (Saddle-to-Saddle)

Usage (run from the project root):

    python analysis/gph_analysis_metrics.py offline
    python analysis/gph_analysis_metrics.py online

Options:
    --recompute       Force recompute statistics (ignore cache)
    --sort-parquet    Sort parquet files by key columns for efficient predicate
                      pushdown (one-time preprocessing step), then exit

Expects sweep outputs in outputs/gph/gph_offline_metrics/ or outputs/gph/gph_online_metrics/.
Saves figures to figures/gph_offline_metrics/ or figures/gph_online_metrics/.
Caches computed statistics in analysis/.cache/gph_offline_metrics.pkl or analysis/.cache/gph_online_metrics.pkl.
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
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from _common import (
    BATCH_KEY_COLS, BL_KEY_COLS, CACHE_DIR, GAMMA_NAMES,
    build_filter, mean_centered_spread, sort_parquet,
)


# =============================================================================
# Configuration
# =============================================================================

METRIC_COLS = [
    "test_loss",
    "grad_norm_squared",
    "trace_gradient_covariance",
]

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph/gph_offline_metrics"),
        "cache_path": CACHE_DIR / "gph_offline_metrics.pkl",
        "figures_path": Path("figures/gph_offline_metrics"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/gph/gph_online_metrics"),
        "cache_path": CACHE_DIR / "gph_online_metrics.pkl",
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
    """Per-metric statistics."""

    mean: np.ndarray       # (n_steps,)
    n: int
    min_vals: np.ndarray | None   # None for deterministic baseline
    max_vals: np.ndarray | None
    spread_lo: np.ndarray | None  # mean-centered 90% band
    spread_hi: np.ndarray | None
    ci_lo: np.ndarray | None      # 95% CI lower (linear-space)
    ci_hi: np.ndarray | None      # 95% CI upper


@dataclass
class ConfigStats:
    """Statistics for a single (width, gamma, noise, model_seed, batch_size) configuration.

    baseline and sgd dicts are keyed by metric name. Keys include METRIC_COLS
    plus "drift_over_diffusion" (computed per-run from drift / diffusion).
    Raw curves (n_runs, n_steps) are stored for individual-run overlay plots.
    """

    steps: np.ndarray
    baseline: dict[str, MetricStats]
    sgd: dict[str, MetricStats]
    baseline_curves: dict[str, np.ndarray] | None = None
    sgd_curves: dict[str, np.ndarray] | None = None


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_metric_curves(df: pl.DataFrame, metric_name: str) -> np.ndarray:
    """Extract metric curves as a (n_runs, n_steps) numpy array."""
    return np.vstack(df[metric_name].to_list())


def _make_metric_stats(curves: np.ndarray) -> MetricStats:
    """Build a MetricStats from a (n_runs, n_steps) array."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1,
            min_vals=None, max_vals=None,
            spread_lo=None, spread_hi=None,
            ci_lo=None, ci_hi=None,
        )

    spread_lo, spread_hi = mean_centered_spread(curves, mean)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(curves.var(axis=0, ddof=1) / n)

    return MetricStats(
        mean=mean, n=n,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
        spread_lo=spread_lo,
        spread_hi=spread_hi,
        ci_lo=mean - t_val * sem,
        ci_hi=mean + t_val * sem,
    )


def _compute_all_metric_stats(
    subset: pl.DataFrame,
) -> tuple[dict[str, MetricStats], dict[str, np.ndarray]]:
    """Compute stats for all metrics including drift/diffusion ratio.

    Returns (metric_stats, raw_curves) where raw_curves maps metric name
    to (n_runs, n_steps) arrays for individual-run plots.
    """
    all_curves = {}
    metric_stats = {}
    for col in METRIC_COLS:
        curves = _extract_metric_curves(subset, col)
        all_curves[col] = curves
        metric_stats[col] = _make_metric_stats(curves)

    drift = all_curves["grad_norm_squared"]
    diffusion = all_curves["trace_gradient_covariance"]
    ratio_curves = drift / np.maximum(diffusion, 1e-30)
    all_curves["drift_over_diffusion"] = ratio_curves
    metric_stats["drift_over_diffusion"] = _make_metric_stats(ratio_curves)

    return metric_stats, all_curves


# =============================================================================
# Statistics Computation
# =============================================================================


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
    baselines: dict[tuple, tuple[np.ndarray, dict[str, MetricStats], dict[str, np.ndarray]]] = {}
    for key, group_df in baseline_groups.items():
        if len(group_df) == 0:
            continue
        steps = np.array(group_df["step"][0])
        bl_stats, bl_curves = _compute_all_metric_stats(group_df)
        baselines[key] = (steps, bl_stats, bl_curves)
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
            sgd_lf.filter(build_filter(BATCH_KEY_COLS, batch_key))
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
            bl_steps, bl_metrics, bl_curves = bl
            sgd_metrics, sgd_curves = _compute_all_metric_stats(group_df)
            key = bl_key + (batch_size,)
            stats[key] = ConfigStats(
                steps=bl_steps,
                baseline=bl_metrics,
                sgd=sgd_metrics,
                baseline_curves=bl_curves,
                sgd_curves=sgd_curves,
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
            "baseline_curves": cs.baseline_curves,
            "sgd_curves": cs.sgd_curves,
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
                baseline_curves=d.get("baseline_curves"),
                sgd_curves=d.get("sgd_curves"),
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


GAMMAS = sorted(GAMMA_NAMES.keys())

ROW_INFO = [
    {"metric": "test_loss", "ylabel": "Test loss", "yscale": "log"},
    {"metric": "grad_norm_squared", "ylabel": "Drift", "yscale": "log"},
    {"metric": "trace_gradient_covariance", "ylabel": "Diffusion", "yscale": "log"},
    {"metric": "drift_over_diffusion", "ylabel": "Drift / Diffusion", "yscale": "log"},
]


def plot_metrics(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    noise: float,
    width: int,
    batch_size: int,
    model_seed: int,
) -> plt.Figure:
    """Metrics: 4 rows (loss, drift, diffusion, drift/diffusion) × 3 gamma columns."""
    baseline_bs = exp_config["baseline_batch_size"]

    # Build labels — loss row uses L_{B=...} notation (matches loss_only script),
    # metric rows use generic batch-size labels since the ylabel names the metric
    if baseline_bs is not None:
        bl_legend_loss = f"$E(L_{{B={baseline_bs}}})$"
        bl_legend_metric = f"$B = {baseline_bs}$"
    else:
        bl_legend_loss = "$L_{\\mathrm{GD}}$"
        bl_legend_metric = "$\\mathrm{GD}$"
    sgd_legend_loss = f"$E(L_{{B={batch_size}}})$"
    sgd_legend_metric = f"$B = {batch_size}$"

    n_rows = len(ROW_INFO)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(15, 10), squeeze=False,
    )

    for col, gamma in enumerate(GAMMAS):
        key = (width, gamma, noise, model_seed, batch_size)
        col_title = f"{GAMMA_NAMES[gamma]} (γ={gamma})"

        # Column headers on top row only
        axes[0, col].set_title(col_title, fontsize=11)

        for row, info in enumerate(ROW_INFO):
            ax = axes[row, col]
            show_ci = info["metric"] != "test_loss"

            if key not in stats:
                if col == 0:
                    ax.set_ylabel(info["ylabel"])
                if row == n_rows - 1:
                    ax.set_xlabel("Training step")
                continue

            s = stats[key]
            bl_ms = s.baseline[info["metric"]]
            sgd_ms = s.sgd[info["metric"]]
            is_loss = info["metric"] == "test_loss"
            bl_label = bl_legend_loss if is_loss else bl_legend_metric
            sgd_label = sgd_legend_loss if is_loss else sgd_legend_metric

            ax.plot(s.steps, bl_ms.mean, label=bl_label, color="C0", linewidth=1.5)

            ax.plot(s.steps, sgd_ms.mean, label=sgd_label, color="C1", linewidth=1.5)
            if show_ci and sgd_ms.ci_lo is not None:
                ax.fill_between(
                    s.steps, sgd_ms.ci_lo, sgd_ms.ci_hi,
                    alpha=0.3, color="C1", label="95% CI",
                )

            ax.set_yscale(info["yscale"])
            if info["yscale"] == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if col == 0:
                ax.set_ylabel(info["ylabel"])
            if row == n_rows - 1:
                ax.set_xlabel("Training step")
            ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    return fig


def plot_metrics_runs(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    noise: float,
    width: int,
    batch_size: int,
    model_seed: int,
) -> plt.Figure | None:
    """Individual-run overlay: thin transparent curves per run, black mean line.

    Same layout as plot_metrics (4 rows × 3 gamma columns) but shows every
    run instead of CI bands. Returns None if no raw curves are available.
    """
    baseline_bs = exp_config["baseline_batch_size"]

    if baseline_bs is not None:
        bl_legend = f"$B = {baseline_bs}$ (mean)"
        sgd_legend = f"$B = {batch_size}$ (mean)"
    else:
        bl_legend = "$\\mathrm{GD}$ (mean)"
        sgd_legend = f"$B = {batch_size}$ (mean)"

    n_rows = len(ROW_INFO)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 10), squeeze=False)

    for col, gamma in enumerate(GAMMAS):
        key = (width, gamma, noise, model_seed, batch_size)
        axes[0, col].set_title(f"{GAMMA_NAMES[gamma]} (γ={gamma})", fontsize=11)

        for row, info in enumerate(ROW_INFO):
            ax = axes[row, col]
            metric = info["metric"]

            if key not in stats:
                if col == 0:
                    ax.set_ylabel(info["ylabel"])
                if row == n_rows - 1:
                    ax.set_xlabel("Training step")
                continue

            s = stats[key]
            if s.sgd_curves is None:
                return None

            bl_ms = s.baseline[metric]
            sgd_ms = s.sgd[metric]
            bl_curves = s.baseline_curves[metric] if s.baseline_curves else None
            sgd_curves = s.sgd_curves[metric]

            # Individual baseline runs
            if bl_curves is not None and len(bl_curves) > 1:
                for curve in bl_curves:
                    ax.plot(s.steps, curve, color="C0", alpha=0.2, linewidth=0.5)
            # Individual SGD runs
            for curve in sgd_curves:
                ax.plot(s.steps, curve, color="C1", alpha=0.2, linewidth=0.5)

            # Mean lines (black, on top)
            ax.plot(s.steps, bl_ms.mean, label=bl_legend, color="black",
                    linewidth=1.5, linestyle="--")
            ax.plot(s.steps, sgd_ms.mean, label=sgd_legend, color="black",
                    linewidth=1.5)

            ax.set_yscale(info["yscale"])
            if info["yscale"] == "log":
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if col == 0:
                ax.set_ylabel(info["ylabel"])
            if row == n_rows - 1:
                ax.set_xlabel("Training step")
            ax.legend(loc="upper right", fontsize=7)

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
    noise, width, batch_size, model_seed, out_dir, filename, plot_type = task
    if plot_type == "ci":
        fig = plot_metrics(
            _worker_ctx["exp"], _worker_ctx["stats"],
            noise, width, batch_size, model_seed,
        )
    else:
        fig = plot_metrics_runs(
            _worker_ctx["exp"], _worker_ctx["stats"],
            noise, width, batch_size, model_seed,
        )
        if fig is None:
            return
    fig.savefig(Path(out_dir) / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_pool(tasks: list[tuple], stats: dict, exp_config: dict, max_workers: int = 0) -> None:
    """Run plot tasks in a multiprocessing pool."""
    if max_workers <= 0:
        max_workers = os.cpu_count() or 1
    n_workers = min(max_workers, len(tasks))
    with Pool(n_workers, initializer=_init_plot_worker, initargs=(stats, exp_config)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_plot_task, tasks), 1):
            print(
                f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="", flush=True,
            )
    print()


def generate_all_plots(exp_config: dict, stats: dict[tuple, ConfigStats]) -> None:
    figures_path = exp_config["figures_path"]
    runs_path = Path(str(figures_path) + "_runs")
    figures_path.mkdir(parents=True, exist_ok=True)
    runs_path.mkdir(parents=True, exist_ok=True)

    # Derive parameter values from stats keys: (width, gamma, noise, model_seed, batch_size)
    widths = sorted({k[0] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    ci_tasks = []
    runs_tasks = []
    for width in widths:
        for noise in noise_levels:
            for model_seed in model_seeds:
                for batch_size in batch_sizes:
                    name = f"drift_diffusion_w{width}_noise{noise}_mseed{model_seed}_b{batch_size}"
                    ci_tasks.append((
                        noise, width, batch_size, model_seed,
                        str(figures_path), name, "ci",
                    ))
                    runs_tasks.append((
                        noise, width, batch_size, model_seed,
                        str(runs_path), name, "runs",
                    ))

    print(f"Generating {len(ci_tasks)} CI figures...")
    _run_pool(ci_tasks, stats, exp_config)

    # Runs plots use fewer workers to avoid OOM (raw curves increase memory per worker)
    print(f"Generating {len(runs_tasks)} individual-runs figures...")
    _run_pool(runs_tasks, stats, exp_config, max_workers=4)

    print(f"All plots saved to {figures_path}/ and {runs_path}/")


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
