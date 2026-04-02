"""
LR Sweep Analysis — Online training with power-law teacher.

One figure per batch size. Each figure is a 3×3 grid:
  rows = gamma values (0.75, 1.0, 1.5)
  cols = model seeds (0, 1, 2)
  each subplot = loss curves for all learning rates

Usage:
    python analysis/lr_sweep.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl

from _common import GAMMA_NAMES
from _lr_sweep_common import DATA_PATH, FIGURES_PATH


def main():
    df = pl.scan_parquet(DATA_PATH).select([
        "model.gamma", "model.model_seed", "training.batch_size",
        "training.lr", "step", "test_loss",
    ]).collect()

    gammas = sorted(df["model.gamma"].unique().to_list())
    model_seeds = sorted(df["model.model_seed"].unique().to_list())
    batch_sizes = sorted(df["training.batch_size"].unique().to_list(), reverse=True)
    lrs = sorted(df["training.lr"].unique().to_list())

    # Color map: log-spaced LRs get a sequential colormap
    cmap = plt.cm.winter
    lr_colors = {lr: cmap(i / (len(lrs) - 1)) for i, lr in enumerate(lrs)}

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for bs in batch_sizes:
        fig, axes = plt.subplots(
            len(gammas), len(model_seeds),
            figsize=(5 * len(model_seeds), 4 * len(gammas)),
            squeeze=False,
        )

        for row, gamma in enumerate(gammas):
            for col, seed in enumerate(model_seeds):
                ax = axes[row, col]
                subset = df.filter(
                    (pl.col("training.batch_size") == bs)
                    & (pl.col("model.gamma") == gamma)
                    & (pl.col("model.model_seed") == seed)
                ).sort("training.lr")

                for lr_row in subset.iter_rows(named=True):
                    lr = lr_row["training.lr"]
                    steps = lr_row["step"]
                    loss = lr_row["test_loss"]
                    ax.plot(
                        steps, loss,
                        color=lr_colors[lr],
                        linewidth=1.2,
                        alpha=0.85,
                        label=f"{lr:.1e}" if col == 0 and row == 0 else None,
                    )

                ax.set_yscale("log")
                if row == 0:
                    ax.set_title(f"Model Seed {seed}", fontsize=11)
                if col == 0:
                    ax.set_ylabel(
                        f"{GAMMA_NAMES[gamma]} (γ={gamma})\nTest Loss",
                        fontsize=10,
                    )
                if row == len(gammas) - 1:
                    ax.set_xlabel("Training Step", fontsize=10)

        # Shared legend from first subplot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Learning Rate",
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            title_fontsize=9,
        )

        fig.suptitle(
            f"LR Sweep — Online Power-Law Teacher | Batch Size = {bs}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])

        out_path = FIGURES_PATH / f"lr_sweep_bs{bs}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nAll plots saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
