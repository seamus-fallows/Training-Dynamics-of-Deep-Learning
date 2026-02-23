"""
Rank metrics analysis: visualize loss and numerical rank evolution during training.

Produces one figure per hidden_dim with two rows:
  - Top: test loss (spanning full width)
  - Bottom: 5 rank metric panels
  - Lines: each model_seed as a thin transparent line, colored by batch_size

Usage:

    python analysis/rank_analysis.py
    python analysis/rank_analysis.py --input outputs/rank_sweep --output figures/rank_sweep
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.cm as cm
import polars as pl
from omegaconf import OmegaConf

from dln.results_io import load_sweep


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT = Path("outputs/rank_sweep")
OUTPUT_PATH = Path("figures/rank_sweep")

METRICS = [
    "absolute_rank",
    "relative_rank",
    "stable_rank",
    "spectral_entropy_rank",
    "participation_ratio",
]

METRIC_LABELS = {
    "absolute_rank": "Absolute Rank",
    "relative_rank": "Relative Rank",
    "stable_rank": "Stable Rank",
    "spectral_entropy_rank": "Spectral Entropy Rank",
    "participation_ratio": "Participation Ratio",
}

# batch_size -> (color, label)
BATCH_STYLES = {
    None: ("#1f77b4", "GD (full batch)"),
    1:    ("#d62728", "SGD (B=1)"),
    5:    ("#ff7f0e", "SGD (B=5)"),
    50:   ("#2ca02c", "SGD (B=50)"),
}

LINE_ALPHA = 0.25
LINE_WIDTH = 0.7


# =============================================================================
# Data Loading
# =============================================================================


def _get_column(row: dict, col: str, defaults: dict):
    """Get a value from a row, falling back to config defaults for non-swept params."""
    if col in row:
        return row[col]
    return defaults.get(col)


def load_results(sweep_dir: Path) -> dict[tuple, list[dict]]:
    """Load sweep results, grouped by (hidden_dim, batch_size).

    Returns dict mapping (hidden_dim, batch_size) -> list of history dicts.
    Each history dict has keys: "step", "test_loss", "absolute_rank", etc.
    """
    df = load_sweep(sweep_dir)

    # Resolve defaults from config for params that weren't swept
    defaults = {}
    config_path = sweep_dir / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        defaults = {
            "model.hidden_dim": cfg.model.hidden_dim,
            "training.batch_size": cfg.training.batch_size,
            "model.model_seed": cfg.model.model_seed,
        }

    list_cols = [c for c in df.columns if isinstance(df[c].dtype, pl.List)]

    results = defaultdict(list)
    for row in df.iter_rows(named=True):
        hidden_dim = _get_column(row, "model.hidden_dim", defaults)
        batch_size = _get_column(row, "training.batch_size", defaults)

        history = {col: row[col] for col in list_cols}
        history["_model_seed"] = _get_column(row, "model.model_seed", defaults)
        results[(hidden_dim, batch_size)].append(history)

    return results


# =============================================================================
# Plotting
# =============================================================================


def _plot_lines(ax, results, hidden_dim, metric_name):
    """Plot all batch sizes for one (hidden_dim, metric) panel."""
    has_data = False

    for batch_size in BATCH_STYLES:
        key = (hidden_dim, batch_size)
        data_list = results.get(key, [])
        if not data_list:
            continue

        color, _ = BATCH_STYLES[batch_size]
        for history in data_list:
            steps = history["step"]
            values = history.get(metric_name)
            if values is None:
                continue
            ax.plot(
                steps, values,
                color=color,
                alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH,
                rasterized=True,
            )
            has_data = True

    if not has_data:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center", transform=ax.transAxes,
        )

    ax.grid(True, alpha=0.3)


def make_figure(results, hidden_dim):
    """Create figure with loss on top (full width) and 5 rank panels below."""
    n_metrics = len(METRICS)
    fig = plt.figure(figsize=(4 * n_metrics, 8), layout="constrained")
    gs = gridspec.GridSpec(2, n_metrics, figure=fig, height_ratios=[1, 1])

    # Top row: test loss spanning all columns
    ax_loss = fig.add_subplot(gs[0, :])
    _plot_lines(ax_loss, results, hidden_dim, "test_loss")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Test Loss")
    ax_loss.set_xlabel("Step")

    handles = [
        mlines.Line2D([], [], color=color, linewidth=2, label=label)
        for color, label in BATCH_STYLES.values()
    ]
    ax_loss.legend(handles=handles, fontsize=9, ncol=len(BATCH_STYLES))

    # Bottom row: rank metrics
    for col, metric_name in enumerate(METRICS):
        ax = fig.add_subplot(gs[1, col], sharex=ax_loss)
        _plot_lines(ax, results, hidden_dim, metric_name)
        ax.set_title(METRIC_LABELS[metric_name], fontsize=10)
        ax.set_xlabel("Step")

    fig.suptitle(
        f"h={hidden_dim} — Loss & Rank Metrics (γ=1.5, no noise)",
        fontsize=14, fontweight="bold",
    )
    return fig


def make_loss_and_relative_rank_figure(results, hidden_dim):
    """Create figure with just loss and relative rank stacked vertically."""
    fig, (ax_loss, ax_rank) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
    )

    _plot_lines(ax_loss, results, hidden_dim, "test_loss")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Test Loss")

    handles = [
        mlines.Line2D([], [], color=color, linewidth=2, label=label)
        for color, label in BATCH_STYLES.values()
    ]
    ax_loss.legend(handles=handles, fontsize=9, ncol=2)

    _plot_lines(ax_rank, results, hidden_dim, "relative_rank")
    ax_rank.set_ylabel("Relative Rank")
    ax_rank.set_xlabel("Step")

    fig.suptitle(
        f"h={hidden_dim} — Loss & Relative Rank (γ=1.5, no noise)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def make_singular_values_figure(results, hidden_dim):
    """Create figure with loss on top and singular value panels per batch_size below.

    Layout: top row = loss (full width), bottom row = one panel per batch_size.
    Each panel shows all singular values colored by index (dark = large, light = small),
    with one thin line per model_seed.
    """
    batch_sizes = [bs for bs in BATCH_STYLES if results.get((hidden_dim, bs))]
    if not batch_sizes:
        return None

    n_panels = len(batch_sizes)
    fig = plt.figure(figsize=(5 * n_panels, 8), layout="constrained")
    gs = gridspec.GridSpec(2, n_panels, figure=fig, height_ratios=[1, 1.2])

    # Top row: loss spanning all columns
    ax_loss = fig.add_subplot(gs[0, :])
    _plot_lines(ax_loss, results, hidden_dim, "test_loss")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Test Loss")
    ax_loss.set_xlabel("Step")

    handles = [
        mlines.Line2D([], [], color=color, linewidth=2, label=label)
        for bs in batch_sizes
        for color, label in [BATCH_STYLES[bs]]
    ]
    ax_loss.legend(handles=handles, fontsize=9, ncol=len(batch_sizes))

    # Determine number of singular values from first available run
    first_run = results[next(k for k in results if k[0] == hidden_dim)][0]
    n_sv = len([k for k in first_run if k.startswith("sv_")])
    cmap = cm.viridis

    # Bottom row: one panel per batch_size
    for col, batch_size in enumerate(batch_sizes):
        ax = fig.add_subplot(gs[1, col], sharex=ax_loss)
        key = (hidden_dim, batch_size)
        data_list = results.get(key, [])

        for history in data_list:
            steps = history["step"]
            for i in range(n_sv):
                values = history.get(f"sv_{i}")
                if values is None:
                    continue
                ax.plot(
                    steps, values,
                    color=cmap(i / max(n_sv - 1, 1)),
                    alpha=LINE_ALPHA,
                    linewidth=LINE_WIDTH,
                    rasterized=True,
                )

        _, label = BATCH_STYLES[batch_size]
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Step")
        if col == 0:
            ax.set_ylabel("Singular Value")
        ax.grid(True, alpha=0.3)

    # Colorbar legend for sv index
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_sv - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes[-n_panels:], shrink=0.6, pad=0.02)
    cbar.set_label("SV index (0 = largest)")

    fig.suptitle(
        f"h={hidden_dim} — Singular Values (γ=1.5, no noise)",
        fontsize=14, fontweight="bold",
    )
    return fig


SEED_GRID_SEEDS = [0, 1, 2, 3, 4]


def _get_n_sv(results, hidden_dim):
    """Get the number of singular value columns from the first available run."""
    for key, data_list in results.items():
        if key[0] == hidden_dim and data_list:
            return len([k for k in data_list[0] if k.startswith("sv_")])
    return 0


def _plot_sv_panel(ax, results, hidden_dim, seed, n_sv, sv_scale):
    """Plot scaled singular values for one (hidden_dim, seed) panel."""
    has_data = False
    for batch_size in BATCH_STYLES:
        key = (hidden_dim, batch_size)
        color, _ = BATCH_STYLES[batch_size]
        for history in results.get(key, []):
            if history.get("_model_seed") != seed:
                continue
            steps = history["step"]
            for i in range(n_sv):
                values = history.get(f"sv_{i}")
                if values is None:
                    continue
                ax.plot(
                    steps, [v * sv_scale for v in values],
                    color=color, linewidth=1.2, rasterized=True,
                )
                has_data = True
    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)


def _plot_seed_panel(ax, results, hidden_dim, seed, metric_name):
    """Plot a single metric for one (hidden_dim, seed) panel."""
    has_data = False
    for batch_size in BATCH_STYLES:
        key = (hidden_dim, batch_size)
        color, _ = BATCH_STYLES[batch_size]
        for history in results.get(key, []):
            if history.get("_model_seed") != seed:
                continue
            steps = history["step"]
            values = history.get(metric_name)
            if values is None:
                continue
            ax.plot(steps, values, color=color, linewidth=1.2, rasterized=True)
            has_data = True
    if not has_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)


def make_seed_grid_figure(results, hidden_dim, rank_metric, sv_scale_value):
    """3 rows × 5 cols: loss / rank metric / scaled singular values, one col per seed."""
    seeds = SEED_GRID_SEEDS
    n_sv = _get_n_sv(results, hidden_dim)

    fig, axes = plt.subplots(
        3, len(seeds), figsize=(4 * len(seeds), 9),
        sharex=True, sharey="row", layout="constrained",
    )

    for col, seed in enumerate(seeds):
        # Row 0: loss
        ax = axes[0, col]
        _plot_seed_panel(ax, results, hidden_dim, seed, "test_loss")
        ax.set_yscale("log")
        ax.set_title(f"seed {seed}", fontsize=10)

        # Row 1: rank metric
        ax = axes[1, col]
        _plot_seed_panel(ax, results, hidden_dim, seed, rank_metric)

        # Row 2: singular values
        ax = axes[2, col]
        _plot_sv_panel(ax, results, hidden_dim, seed, n_sv, sv_scale_value)
        ax.set_xlabel("Step")

    axes[0, 0].set_ylabel("Test Loss")
    axes[1, 0].set_ylabel(METRIC_LABELS[rank_metric])
    axes[2, 0].set_ylabel(f"SV / {1/sv_scale_value:.0f}" if sv_scale_value != 1.0
                           else "Singular Value")

    handles = [
        mlines.Line2D([], [], color=color, linewidth=2, label=label)
        for color, label in BATCH_STYLES.values()
    ]
    axes[0, -1].legend(handles=handles, fontsize=8, loc="upper right")

    return fig


def make_seed_grid_entropy_figure(results, hidden_dim):
    """Seed grid with spectral entropy rank; SVs scaled to match entropy range."""
    seeds = SEED_GRID_SEEDS
    n_sv = _get_n_sv(results, hidden_dim)

    # Compute scaling: map max SV to max observed spectral entropy rank
    max_sv, max_entropy = 0.0, 0.0
    for batch_size in BATCH_STYLES:
        key = (hidden_dim, batch_size)
        for history in results.get(key, []):
            if history.get("_model_seed") not in seeds:
                continue
            entropy_vals = history.get("spectral_entropy_rank")
            if entropy_vals:
                max_entropy = max(max_entropy, max(entropy_vals))
            for i in range(n_sv):
                sv_vals = history.get(f"sv_{i}")
                if sv_vals:
                    max_sv = max(max_sv, max(sv_vals))

    sv_scale = max_entropy / max_sv if max_sv > 0 else 1.0

    fig = make_seed_grid_figure(
        results, hidden_dim, "spectral_entropy_rank", sv_scale,
    )
    axes = fig.axes
    # Fix the y-label for the SV row
    axes[2 * len(seeds)].set_ylabel(f"SV × {sv_scale:.3f}")

    fig.suptitle(
        f"h={hidden_dim} — Loss / Spectral Entropy Rank / Scaled SVs",
        fontsize=14, fontweight="bold",
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Plot rank metrics from sweep results")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Path to save figures",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)

    if not results:
        print("No results found!")
        return

    hidden_dims = sorted({hd for hd, _ in results})

    print(f"Found {len(results)} (hidden_dim, batch_size) groups:")
    for key, data_list in sorted(results.items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        batch_label = "GD" if key[1] is None else f"B={key[1]}"
        print(f"  h={key[0]}, {batch_label}: {len(data_list)} runs")

    args.output.mkdir(parents=True, exist_ok=True)

    for hidden_dim in hidden_dims:
        # All rank metrics figure
        fig = make_figure(results, hidden_dim)
        output_file = args.output / f"rank_metrics_h{hidden_dim}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)

        # Loss + relative rank figure
        fig = make_loss_and_relative_rank_figure(results, hidden_dim)
        output_file = args.output / f"loss_relative_rank_h{hidden_dim}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)

        # Singular values figure
        fig = make_singular_values_figure(results, hidden_dim)
        if fig is not None:
            output_file = args.output / f"singular_values_h{hidden_dim}.png"
            fig.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close(fig)

        # Seed grid: relative rank + SVs/10
        fig = make_seed_grid_figure(
            results, hidden_dim, "relative_rank", sv_scale_value=1.0 / 10,
        )
        fig.suptitle(
            f"h={hidden_dim} — Loss / Relative Rank / SVs÷10",
            fontsize=14, fontweight="bold",
        )
        output_file = args.output / f"seed_grid_relative_rank_h{hidden_dim}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)

        # Seed grid: spectral entropy rank + scaled SVs
        fig = make_seed_grid_entropy_figure(results, hidden_dim)
        output_file = args.output / f"seed_grid_entropy_rank_h{hidden_dim}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
