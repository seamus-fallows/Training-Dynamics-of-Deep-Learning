"""
GPH Loss Analysis — Combined Online/Offline Figures

Generates 4×3 figures comparing online and offline training regimes:
  Row 1: Online loss (baseline + SGD mean + significance shading)
  Row 2: Online loss ratio with CI
  Row 3: Offline loss
  Row 4: Offline loss ratio
  Columns: γ = 0.75 (NTK), γ = 1.0 (Mean-Field), γ = 1.5 (Saddle-to-Saddle)

One figure per (width, noise_std, model_seed, batch_size) combination.

Usage (run from the project root):

    python analysis/gph_analysis_loss_only.py
    python analysis/gph_analysis_loss_only.py --recompute
    python analysis/gph_analysis_loss_only.py --sort-parquet

Data: outputs/gph_offline/ and outputs/gph_online/
Output: figures/gph_loss/
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

from _common import (
    BATCH_KEY_COLS,
    BL_KEY_COLS,
    CACHE_DIR,
    GAMMA_NAMES,
    build_filter,
    mean_centered_spread,
    sort_parquet,
)


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph_offline"),
        "cache_path": CACHE_DIR / "gph_offline_loss.pkl",
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
    },
    "online": {
        "base_path": Path("outputs/gph_online"),
        "cache_path": CACHE_DIR / "gph_online_loss.pkl",
        "baseline_subdir": "large_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
    },
}

FIGURES_PATH = Path("figures/gph_loss")
GAMMAS = sorted(GAMMA_NAMES.keys())


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
    n_runs: int
    # 95% CI — linear-space (arithmetic)
    sgd_ci_lo: np.ndarray
    sgd_ci_hi: np.ndarray
    # Loss ratio
    ratio: np.ndarray
    ratio_ci_lo: np.ndarray
    ratio_ci_hi: np.ndarray
    sig_mask: np.ndarray  # bool: baseline significantly < SGD at 95%


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_curves(df: pl.DataFrame) -> np.ndarray:
    """Extract loss curves as a (n_runs, n_steps) numpy array."""
    return np.vstack(df["test_loss"].to_list())


def _welch_t_crit(
    se_a: np.ndarray,
    n_a: int,
    se_b: np.ndarray,
    n_b: int,
    quantile: float = 0.975,
) -> np.ndarray:
    """Welch-Satterthwaite t critical value, numerically stable.

    se_a, se_b are variance/n (squared standard error of the mean).
    quantile: 0.975 for two-tailed 95% CI, 0.95 for one-sided p<0.05 test.
    When both SEs underflow to zero, uses normal approximation (large df).
    """
    se_sum = se_a + se_b
    ws_numer = se_sum**2
    ws_denom = se_a**2 / max(n_a - 1, 1) + se_b**2 / max(n_b - 1, 1)
    # When ws_denom underflows to 0, SEs are negligible → large df (normal approx)
    nonzero = ws_denom > 0
    df = np.where(nonzero, ws_numer / np.where(nonzero, ws_denom, 1.0), 1e6)
    df = np.maximum(df, 1.0)
    return scipy_stats.t.ppf(quantile, df=df)


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
    spread_lo, spread_hi = mean_centered_spread(curves, mean)

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
    subset: pl.DataFrame,
    bl: BaselineStats,
) -> ConfigStats | None:
    """Compute SGD statistics and precompute all plotting quantities."""
    if len(subset) == 0:
        return None

    curves = _extract_curves(subset)
    n = len(curves)

    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1) if n > 1 else np.zeros(curves.shape[1])
    spread_lo, spread_hi = mean_centered_spread(curves, mean) if n > 1 else (mean, mean)

    # --- Precompute plotting quantities ---

    df_sgd = max(n - 1, 1)
    sem = np.sqrt(var / n)

    # 95% CI for the arithmetic mean (two-tailed)
    t_ci = scipy_stats.t.ppf(0.975, df=df_sgd)
    sgd_ci_lo = mean - t_ci * sem
    sgd_ci_hi = mean + t_ci * sem

    # Loss ratio E(L_SGD) / L_baseline and its CI
    ratio = mean / bl.loss

    if bl.var is None:
        # Deterministic baseline: CI transforms linearly
        ratio_ci_lo = sgd_ci_lo / bl.loss
        ratio_ci_hi = sgd_ci_hi / bl.loss
        # One-sided test: E[SGD] > baseline at p<0.05
        t_sig = scipy_stats.t.ppf(0.95, df=df_sgd)
        sig_mask = bl.loss < (mean - t_sig * sem)
    else:
        # Delta method: Var(Y/X) ≈ (Y/X)² × (Var(Y)/(n_Y·Y²) + Var(X)/(n_X·X²))
        safe_sgd = np.maximum(mean, 1e-30)
        safe_bl = np.maximum(bl.loss, 1e-30)
        rel_var = var / (n * safe_sgd**2) + bl.var / (bl.n * safe_bl**2)
        se_ratio = ratio * np.sqrt(rel_var)

        se_bl = bl.var / bl.n
        se_sgd = var / n
        t_crit = _welch_t_crit(se_bl, bl.n, se_sgd, n)  # two-tailed for CI

        ratio_ci_lo = ratio - t_crit * se_ratio
        ratio_ci_hi = ratio + t_crit * se_ratio

        # One-sided Welch's t-test: E[SGD] - E[baseline] > 0 at p<0.05
        t_sig = _welch_t_crit(se_bl, bl.n, se_sgd, n, quantile=0.95)
        diff = mean - bl.loss
        se_diff = np.sqrt(se_bl + se_sgd)
        sig_mask = diff > t_sig * se_diff

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
    print(
        f"Computing SGD statistics ({n_batches} batches, {total_configs} baseline groups)..."
    )

    stats: dict[tuple, ConfigStats] = {}
    completed = 0
    for batch_key, batch_bl_keys in key_batches.items():
        chunk = (
            sgd_lf.filter(build_filter(BATCH_KEY_COLS, batch_key))
            .select(sgd_select)
            .collect()
        )
        sub_groups = chunk.partition_by(
            ["model.model_seed", "training.batch_size"],
            as_dict=True,
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
    exp_config: dict,
    force_recompute: bool = False,
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


def plot_combined(
    online_stats: dict[tuple, ConfigStats],
    offline_stats: dict[tuple, ConfigStats],
    width: int,
    noise: float,
    model_seed: int,
    batch_size: int,
) -> plt.Figure:
    """Combined online/offline figure: 4 rows × 3 gamma columns.

    Rows: online loss, online ratio, offline loss, offline ratio.
    Columns: NTK (γ=0.75), Mean-Field (γ=1.0), Saddle-to-Saddle (γ=1.5).
    """
    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(2, 1, hspace=0.03)

    # (label, stats_dict, baseline_batch_size)
    regimes = [
        ("Online (i.i.d. samples per step)", online_stats, 500),
        ("Offline (N = 500 fixed samples)", offline_stats, None),
    ]

    for subfig, (regime_label, stats_dict, baseline_bs) in zip(subfigs, regimes):
        axes = subfig.subplots(2, 3, squeeze=False)
        subfig.suptitle(regime_label, fontsize=12, fontweight="bold")

        # Build labels
        if baseline_bs is not None:
            bl_legend = f"$E(L_{{B={baseline_bs}}})$"
            sgd_legend = f"$E(L_{{B={batch_size}}})$"
            ratio_ylabel = f"$E(L_{{B={batch_size}}}) \\,/\\, E(L_{{B={baseline_bs}}})$"
            sig_label = f"$E(L_{{B={baseline_bs}}}) < E(L_{{B={batch_size}}})$ (p<0.05)"
        else:
            bl_legend = "$L_{\\mathrm{GD}}$"
            sgd_legend = f"$E(L_{{B={batch_size}}})$"
            ratio_ylabel = f"$E(L_{{B={batch_size}}}) \\,/\\, L_{{\\mathrm{{GD}}}}$"
            sig_label = f"$L_{{\\mathrm{{GD}}}} < E(L_{{B={batch_size}}})$ (p<0.05)"

        for col, gamma in enumerate(GAMMAS):
            key = (width, gamma, noise, model_seed, batch_size)
            ax_loss = axes[0, col]
            ax_ratio = axes[1, col]

            ax_loss.set_title(f"{GAMMA_NAMES[gamma]} (γ={gamma})", fontsize=11)

            if key not in stats_dict:
                if col == 0:
                    ax_loss.set_ylabel("Test loss")
                    ax_ratio.set_ylabel(ratio_ylabel)
                continue

            s = stats_dict[key]

            # === Loss row ===
            ax_loss.plot(s.steps, s.baseline_loss, label=bl_legend, color="C0", linewidth=1.5)
            ax_loss.plot(s.steps, s.sgd_mean, label=sgd_legend, color="C1", linewidth=1.5)
            ax_loss.fill_between(
                s.steps, 0, 1, where=s.sig_mask, alpha=0.25, color="darkgreen",
                transform=ax_loss.get_xaxis_transform(), label=sig_label,
            )
            ax_loss.set_yscale("log")
            if col == 0:
                ax_loss.set_ylabel("Test loss")
            ax_loss.legend(loc="upper right", fontsize=7)

            # === Ratio row ===
            ax_ratio.axhline(1.0, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
            ax_ratio.plot(s.steps, s.ratio, color="C1", linewidth=1.5)
            ax_ratio.fill_between(
                s.steps, s.ratio_ci_lo, s.ratio_ci_hi,
                alpha=0.3, color="C1", label="95% CI",
            )
            if col == 0:
                ax_ratio.set_ylabel(ratio_ylabel)
            ax_ratio.set_xlabel("Training step")
            ax_ratio.legend(loc="upper right", fontsize=7)

    return fig


# =============================================================================
# Parallel Plot Generation
# =============================================================================

# Module-level state for multiprocessing workers (set by _init_plot_worker)
_worker_ctx: dict = {}


def _init_plot_worker(
    online_stats: dict,
    offline_stats: dict,
) -> None:
    """Initializer for plot worker processes. Called once per worker."""
    _worker_ctx["online"] = online_stats
    _worker_ctx["offline"] = offline_stats


def _run_plot_task(task: tuple) -> None:
    """Worker function: generate one figure and save to disk."""
    width, noise, model_seed, batch_size, out_dir, filename = task
    fig = plot_combined(
        _worker_ctx["online"],
        _worker_ctx["offline"],
        width,
        noise,
        model_seed,
        batch_size,
    )
    fig.savefig(Path(out_dir) / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    online_stats: dict[tuple, ConfigStats],
    offline_stats: dict[tuple, ConfigStats],
) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Derive parameter values from both stats dicts
    all_keys = set(online_stats.keys()) | set(offline_stats.keys())
    widths = sorted({k[0] for k in all_keys})
    noise_levels = sorted({k[2] for k in all_keys})
    model_seeds = sorted({k[3] for k in all_keys})
    batch_sizes = sorted({k[4] for k in all_keys})

    # Build task list — one figure per (width, noise, seed, batch_size)
    tasks = []
    for width in widths:
        for noise in noise_levels:
            for model_seed in model_seeds:
                for batch_size in batch_sizes:
                    filename = f"loss_ratio_w{width}_noise{noise}_mseed{model_seed}_b{batch_size}"
                    tasks.append(
                        (
                            width,
                            noise,
                            model_seed,
                            batch_size,
                            str(FIGURES_PATH),
                            filename,
                        )
                    )

    n_workers = min(6, os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_plot_worker,
        initargs=(online_stats, offline_stats),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_plot_task, tasks), 1):
            print(
                f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="",
                flush=True,
            )

    print(f"\nAll plots saved to {FIGURES_PATH}/")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GPH loss analysis — combined online/offline figures",
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

    if args.sort_parquet:
        for name, exp in EXPERIMENTS.items():
            print(f"Sorting {name} parquet files...")
            sort_parquet(exp)
        print("Done! Re-run without --sort-parquet to analyze.")
        return

    print("Loading online stats...")
    online_stats = get_stats(EXPERIMENTS["online"], args.recompute)
    print("Loading offline stats...")
    offline_stats = get_stats(EXPERIMENTS["offline"], args.recompute)

    generate_all_plots(online_stats, offline_stats)
    print("\nDone!")


if __name__ == "__main__":
    main()
