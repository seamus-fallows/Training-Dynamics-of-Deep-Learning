"""
LR Sweep — Singular Value Dynamics

One figure per (batch_size, seed). Each figure is a 5×3 grid:
  rows = singular value index (sigma_0 through sigma_4)
  cols = gamma values (0.75, 1.0, 1.5)
  each subplot = that SV's trajectory for all learning rates

Y-axis shared within each column so smaller SVs are visually smaller.
Teacher target shown as dashed horizontal line per subplot.

Usage:
    python analysis/lr_sweep_sv.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import GAMMA_NAMES

DATA_PATH = Path("outputs/lr_sweep_online/results.parquet")
FIGURES_PATH = Path("figures/lr_sweep_online")

# Teacher singular values: scale * k^{-alpha} for k=1..5, scale=50, alpha=1
TEACHER_SVS = [50 / k for k in range(1, 6)]
N_SVS = 5


PLOT_SEEDS = [0, 1]


def main():
    df = pl.read_parquet(DATA_PATH)

    gammas = sorted(df["model.gamma"].unique().to_list())
    batch_sizes = sorted(df["training.batch_size"].unique().to_list(), reverse=True)
    lrs = sorted(df["training.lr"].unique().to_list())

    cmap = plt.cm.winter
    lr_colors = {lr: cmap(i / (len(lrs) - 1)) for i, lr in enumerate(lrs)}

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for bs in batch_sizes:
        for seed in PLOT_SEEDS:
            fig, axes = plt.subplots(
                N_SVS, len(gammas),
                figsize=(6 * len(gammas), 3 * N_SVS),
                squeeze=False,
            )

            for col, gamma in enumerate(gammas):
                for row in range(N_SVS):
                    ax = axes[row, col]
                    sv_col = f"sv_{row}"
                    teacher_val = TEACHER_SVS[row]

                    ax.axhline(teacher_val, color="black", ls="--", lw=1.0, alpha=0.4)

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
                        ax.set_ylabel(
                            f"$\\sigma_{row}$  (target={teacher_val:.1f})",
                            fontsize=10,
                        )
                    if row == N_SVS - 1:
                        ax.set_xlabel("Training Step", fontsize=10)

            # Shared y-limits per column
            for col in range(len(gammas)):
                col_ymax = max(axes[row, col].get_ylim()[1] for row in range(N_SVS))
                for row in range(N_SVS):
                    axes[row, col].set_ylim(0, col_ymax)

            fig.suptitle(
                f"End-to-End Singular Values — Online Power-Law Teacher"
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

            out_path = FIGURES_PATH / f"e2e_sv_bs{bs}_seed{seed}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")

    print(f"\nAll plots saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
