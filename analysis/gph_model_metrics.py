"""
GPH Model Metrics Analysis

Plots per-layer ref vs SGD comparison of model-level metrics (gram norms,
balance diffs) for each (width, noise, model_seed, batch_size) combination.

Reads cached statistics produced by gph_comparative_metrics.py — run that
script first if the cache does not exist.

Usage (run from the project root):

    python analysis/gph_model_metrics.py

Output: figures/gph_model_metrics/
"""

import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import CACHE_DIR, GAMMA_NAMES
from gph_comparative_metrics import (
    CompConfigStats,
    MetricStats,
    _load_cache,
)


# =============================================================================
# Configuration
# =============================================================================

FIGURES_PATH = Path("figures/gph_model_metrics")
N_WORKERS = 10


# =============================================================================
# Plotting
# =============================================================================


def _plot_ref_vs_sgd_layers(
    ax: plt.Axes,
    steps: np.ndarray,
    ref_dict: dict[int, MetricStats],
    sgd_dict: dict[int, MetricStats],
    ylabel: str,
    ref_label: str,
    sgd_label: str = "SGD",
    show_legend: bool = False,
) -> None:
    """Plot per-layer ref (solid) vs SGD (dashed + CI) comparison."""
    for i in sorted(ref_dict.keys()):
        color = f"C{i}"
        ref = ref_dict[i]
        ax.plot(
            steps, ref.mean, color=color, linewidth=1.5,
            label=f"L{i} {ref_label}",
        )
        if ref.n > 1:
            ax.fill_between(
                steps, ref.ci_lo, ref.ci_hi, alpha=0.1, color=color,
            )
        if i in sgd_dict:
            sgd = sgd_dict[i]
            ax.plot(
                steps, sgd.mean, color=color, linewidth=1.5,
                linestyle="--", label=f"L{i} {sgd_label}",
            )
            ax.fill_between(
                steps, sgd.ci_lo, sgd.ci_hi, alpha=0.15, color=color,
            )
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if show_legend:
        ax.legend(fontsize=7, loc="best", ncols=2)


def plot_model_metrics(
    stats: dict[tuple, CompConfigStats],
    model_seed: int,
    noise: float,
    width: int,
    batch_size: int,
    gammas: list[float],
    ref_label: str,
) -> plt.Figure:
    """2 rows: gram norms, balance diffs. Columns = gamma."""
    n_cols = len(gammas)
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    for col, gamma in enumerate(gammas):
        key = (width, gamma, noise, model_seed, batch_size)
        axes[0, col].set_title(GAMMA_NAMES.get(gamma, str(gamma)), fontsize=10)
        if key not in stats:
            continue

        s = stats[key]
        last_col = col == n_cols - 1
        sgd_lbl = f"B={batch_size}"

        # Row 0: Gram norms
        _plot_ref_vs_sgd_layers(
            axes[0, col], s.steps, s.ref_gram_norms, s.sgd_gram_norms,
            ylabel=r"$\|W_i W_i^T\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_lbl, show_legend=last_col,
        )

        # Row 1: Balance diffs
        _plot_ref_vs_sgd_layers(
            axes[1, col], s.steps, s.ref_balance_diffs, s.sgd_balance_diffs,
            ylabel=r"$\|G_l\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_lbl, show_legend=last_col,
        )
        axes[1, col].set_xlabel("Training step")

    fig.tight_layout()
    return fig


# =============================================================================
# Parallel Plot Generation
# =============================================================================

_worker_ctx: dict = {}


def _init_worker(all_stats: dict, output_dir: str) -> None:
    _worker_ctx["stats"] = all_stats
    _worker_ctx["output_dir"] = output_dir


def _run_task(task: tuple) -> None:
    subdir, seed, noise, width, bsize, gammas, ref_lbl, fname = task
    fig = plot_model_metrics(
        _worker_ctx["stats"], seed, noise, width, bsize, gammas, ref_lbl,
    )
    if fig is None:
        return
    dest = Path(_worker_ctx["output_dir"]) / subdir
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    stats: dict[tuple, CompConfigStats],
    output_dir: Path,
    ref_label: str,
    regime_name: str,
) -> None:
    regime = regime_name.lower()

    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    has_sgd_metrics = any(len(cs.sgd_gram_norms) > 0 for cs in stats.values())
    if not has_sgd_metrics:
        print(f"  Note: SGD model metrics not available for {regime_name} — skipping.")
        return

    tasks = []
    for model_seed in model_seeds:
        for noise in noise_levels:
            subdir = f"{regime}/noise_{noise}"
            for width in widths:
                tag = f"seed{model_seed}_w{width}"
                for batch_size in batch_sizes:
                    tasks.append((
                        subdir, model_seed, noise, width,
                        batch_size, gammas, ref_label,
                        f"model_metrics_{tag}_b{batch_size}",
                    ))

    n_workers = min(N_WORKERS, len(tasks))
    print(f"Generating {len(tasks)} {regime_name} figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(stats, str(output_dir)),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            if i % 5 == 0 or i == len(tasks):
                print(
                    f"\r  {regime_name}: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                    end="", flush=True,
                )

    print(f"\n{regime_name} plots saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================


def main():
    offline_cache = CACHE_DIR / "gph_combined_offline.pkl"
    online_cache = CACHE_DIR / "gph_combined_online.pkl"

    if offline_cache.exists():
        print("=== Offline Regime ===")
        offline_stats = _load_cache(offline_cache)
        if offline_stats:
            generate_all(offline_stats, FIGURES_PATH, ref_label="GD", regime_name="Offline")
        else:
            print("  Failed to load offline cache.")
    else:
        print(f"Offline cache not found at {offline_cache} — run gph_comparative_metrics.py first.")

    if online_cache.exists():
        print("\n=== Online Regime ===")
        online_stats = _load_cache(online_cache)
        if online_stats:
            generate_all(online_stats, FIGURES_PATH, ref_label="B=500", regime_name="Online")
        else:
            print("  Failed to load online cache.")
    else:
        print(f"Online cache not found at {online_cache} — run gph_comparative_metrics.py first.")

    print("\nDone!")


if __name__ == "__main__":
    main()
