"""
LR Sweep — Per-Layer Singular Value Dynamics

One figure per (batch_size, seed, layer). Each figure is a 5×3 grid:
  rows = singular value index (top 5)
  cols = gamma values (0.75, 1.0, 1.5)
  each subplot = that SV's trajectory for all learning rates

Y-axis shared within each column so smaller SVs are visually smaller.

Usage:
    python analysis/lr_sweep_layer_sv.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import GAMMA_NAMES
from _lr_sweep_common import DATA_PATH, FIGURES_PATH

# Architecture: [5, 100, 100, 100, 5] -> 4 layers
LAYER_SHAPES = [(100, 5), (100, 100), (100, 100), (5, 100)]
N_PLOT_SVS = 5


N_LAYERS = len(LAYER_SHAPES)

PLOT_SEEDS = [0, 1]


def main():
    # Build list of layer SV columns needed
    layer_sv_cols = [
        f"layer_{layer_idx}_sv_{sv_idx}"
        for layer_idx in range(N_LAYERS)
        for sv_idx in range(N_PLOT_SVS)
    ]
    df = pl.scan_parquet(DATA_PATH).select([
        "model.gamma", "model.model_seed", "training.batch_size",
        "training.lr", "step",
    ] + layer_sv_cols).collect()

    gammas = sorted(df["model.gamma"].unique().to_list())
    batch_sizes = sorted(df["training.batch_size"].unique().to_list(), reverse=True)
    lrs = sorted(df["training.lr"].unique().to_list())

    cmap = plt.cm.winter
    lr_colors = {lr: cmap(i / (len(lrs) - 1)) for i, lr in enumerate(lrs)}

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for bs in batch_sizes:
        for seed in PLOT_SEEDS:
            for layer_idx in range(N_LAYERS):
                out_rows, out_cols = LAYER_SHAPES[layer_idx]
                n_svs = min(out_rows, out_cols)
                n_plot = min(N_PLOT_SVS, n_svs)

                fig, axes = plt.subplots(
                    n_plot, len(gammas),
                    figsize=(6 * len(gammas), 3 * n_plot),
                    squeeze=False,
                )

                for col, gamma in enumerate(gammas):
                    for row in range(n_plot):
                        ax = axes[row, col]
                        sv_col = f"layer_{layer_idx}_sv_{row}"

                        subset = df.filter(
                            (pl.col("training.batch_size") == bs)
                            & (pl.col("model.gamma") == gamma)
                            & (pl.col("model.model_seed") == seed)
                        ).sort("training.lr")

                        for lr_row in subset.iter_rows(named=True):
                            lr = lr_row["training.lr"]
                            label = (
                                f"{lr:.1e}" if row == 0 and col == len(gammas) - 1
                                else None
                            )
                            ax.plot(
                                lr_row["step"], lr_row[sv_col],
                                color=lr_colors[lr], linewidth=1.2, alpha=0.85,
                                label=label,
                            )

                        if row == 0:
                            ax.set_title(
                                f"{GAMMA_NAMES[gamma]} ($\\gamma$={gamma})",
                                fontsize=11,
                            )
                        if col == 0:
                            ax.set_ylabel(f"$\\sigma_{row}$", fontsize=10)
                        if row == n_plot - 1:
                            ax.set_xlabel("Training Step", fontsize=10)

                # Shared y-limits per column
                for col in range(len(gammas)):
                    col_ymax = max(
                        axes[row, col].get_ylim()[1] for row in range(n_plot)
                    )
                    for row in range(n_plot):
                        axes[row, col].set_ylim(0, col_ymax)

                layer_label = (
                    f"$W_{layer_idx}$  ({out_rows}$\\times${out_cols})"
                    f"{'  [top ' + str(n_plot) + ' SVs]' if n_plot < n_svs else ''}"
                )
                fig.suptitle(
                    f"Layer {layer_idx} Singular Values — {layer_label}"
                    f" | Batch Size = {bs}, Seed = {seed}",
                    fontsize=14,
                    fontweight="bold",
                )
                handles, labels = axes[0, -1].get_legend_handles_labels()
                fig.legend(
                    handles, labels,
                    title="Learning Rate",
                    loc="center right",
                    bbox_to_anchor=(1.0, 0.5),
                    fontsize=8,
                    title_fontsize=9,
                )
                fig.tight_layout(rect=[0, 0, 0.92, 0.96])

                out_path = FIGURES_PATH / f"layer{layer_idx}_sv_bs{bs}_seed{seed}.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out_path}")

    print(f"\nAll plots saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
