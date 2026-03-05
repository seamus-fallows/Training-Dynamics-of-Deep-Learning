"""
LR Sweep — Loss Comparison: Large vs Small Batch Size

One figure per (small_batch_size, online_mode). Each figure has:
  rows = 6 learning rates (the grid for the small batch size)
  columns = gamma values (0.75, 1.0, 1.5)
  each subplot = test loss for Batch size 500 (dashed) and small (solid) overlaid

Usage:
    python analysis/lr_sweep_loss_compare.py
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

LARGE_BS = 500
SMALL_BS_LIST = [50, 5, 1]

# Per-batch-size LR grids (6 log-spaced from 1e-4 to max stable)
LR_GRIDS = {
    50:  np.logspace(np.log10(0.0001), np.log10(0.0041), 6),
    5:   np.logspace(np.log10(0.0001), np.log10(0.003), 6),
    1:   np.logspace(np.log10(0.0001), np.log10(0.001), 6),
}


def _closest_lr(target, available):
    return min(available, key=lambda x: abs(x - target))


def _bs_label(bs, online):
    if not online and bs == LARGE_BS:
        return "GD"
    return f"Batch size {bs}"


def main():
    df = pl.read_parquet(DATA_PATH)
    gammas = sorted(df["model.gamma"].unique().to_list())
    available_lrs = sorted(df["training.lr"].unique().to_list())
    online_modes = sorted(df["data.online"].unique().to_list(), reverse=True)

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for online in online_modes:
        mode_label = "Online" if online else "Offline"

        for small_bs in SMALL_BS_LIST:
            target_lrs = LR_GRIDS[small_bs]
            lrs = [_closest_lr(t, available_lrs) for t in target_lrs]
            seen = set()
            lrs = [lr for lr in lrs if not (lr in seen or seen.add(lr))]
            n_lrs = len(lrs)

            fig, axes = plt.subplots(
                n_lrs, len(gammas),
                figsize=(6 * len(gammas), 3 * n_lrs),
                squeeze=False,
            )

            large_label = _bs_label(LARGE_BS, online)
            small_label = _bs_label(small_bs, online)

            for gi, gamma in enumerate(gammas):
                for row, lr in enumerate(lrs):
                    ax = axes[row, gi]

                    large = df.filter(
                        (pl.col("training.batch_size") == LARGE_BS)
                        & (pl.col("model.gamma") == gamma)
                        & (pl.col("data.online") == online)
                        & (pl.col("training.lr") == lr)
                    )
                    small = df.filter(
                        (pl.col("training.batch_size") == small_bs)
                        & (pl.col("model.gamma") == gamma)
                        & (pl.col("data.online") == online)
                        & (pl.col("training.lr") == lr)
                    )

                    if large.height > 0:
                        large_row = large.row(0, named=True)
                        ax.plot(
                            large_row["step"], large_row["test_loss"],
                            ls="--", linewidth=1.2,
                            label=large_label if row == 0 and gi == 0 else None,
                        )
                    if small.height > 0:
                        small_row = small.row(0, named=True)
                        ax.plot(
                            small_row["step"], small_row["test_loss"],
                            linewidth=1.2,
                            label=small_label if row == 0 and gi == 0 else None,
                        )
                    ax.set_yscale("log")

                    if row == 0:
                        ax.set_title(
                            f"{GAMMA_NAMES[gamma]} ($\\gamma$={gamma})",
                            fontsize=10,
                        )
                    if gi == 0:
                        ax.set_ylabel(f"lr={lr:.1e}", fontsize=9)
                    if row == 0 and gi == 0:
                        ax.legend(fontsize=8, loc="upper right")
                    if row == n_lrs - 1:
                        ax.set_xlabel("Step", fontsize=9)

            fig.suptitle(
                f"Test Loss: {large_label} vs {small_label}"
                f" | {mode_label} | Power-Law Teacher",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = FIGURES_PATH / f"loss_compare_bs{small_bs}_{mode_label.lower()}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")

    print(f"\nAll plots saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
