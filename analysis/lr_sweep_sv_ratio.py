"""
LR Sweep — Singular Value Comparison: Large vs Small Batch Size

One figure per (partial_product, small_batch_size, online_mode). Each figure:
  rows = 6 learning rates (the grid for the small batch size)
  columns = gamma × 2 (raw SVs | SV ratio large/small)
    Left: raw SVs, solid = small batch, dashed = large batch (GD in offline)
    Right: ratio large/small for each SV index

Colors = SV index (tab10), top 5.

With 10 partial products × 3 small batch sizes × 2 modes = 60 figures.

Usage:
    python analysis/lr_sweep_sv_ratio.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# All partial products for 4-layer network (indices 0-3)
PARTIAL_PRODUCTS = []
for i in range(4):
    for j in range(i, 4):
        PARTIAL_PRODUCTS.append((i, j))

LAYER_DIMS = [5, 100, 100, 100, 5]


def _pp_label(i, j):
    if i == j:
        rows, cols = LAYER_DIMS[j + 1], LAYER_DIMS[j]
        return f"$W_{i}$ ({rows}×{cols})"
    return f"$P({i},{j})$ = $W_{j} \\cdots W_{i}$"


def _pp_shape(i, j):
    return (LAYER_DIMS[j + 1], LAYER_DIMS[i])


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

    sv_cmap = plt.cm.tab10

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    for online in online_modes:
        mode_label = "Online" if online else "Offline"

        for small_bs in SMALL_BS_LIST:
            target_lrs = LR_GRIDS[small_bs]
            lrs = [_closest_lr(t, available_lrs) for t in target_lrs]
            seen = set()
            lrs = [lr for lr in lrs if not (lr in seen or seen.add(lr))]
            n_lrs = len(lrs)

            large_label = _bs_label(LARGE_BS, online)
            small_label = _bs_label(small_bs, online)
            n_cols = len(gammas) * 2  # raw + ratio per gamma

            for pp_i, pp_j in PARTIAL_PRODUCTS:
                sv_col = f"pp_{pp_i}_{pp_j}_sv"
                rows_out, cols_out = _pp_shape(pp_i, pp_j)
                n_svs = min(rows_out, cols_out)
                n_plot_svs = min(n_svs, 5)

                fig, axes = plt.subplots(
                    n_lrs, n_cols,
                    figsize=(5 * n_cols, 3 * n_lrs),
                    squeeze=False,
                )

                for gi, gamma in enumerate(gammas):
                    col_raw = gi * 2
                    col_ratio = gi * 2 + 1

                    for row, lr in enumerate(lrs):
                        ax_raw = axes[row, col_raw]
                        ax_rat = axes[row, col_ratio]

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

                        svs_large_arr = None
                        svs_small_arr = None
                        steps_large = None
                        steps_small = None

                        if large.height > 0:
                            large_row = large.row(0, named=True)
                            steps_large = np.array(large_row["step"])
                            svs_large_arr = np.array(large_row[sv_col])

                        if small.height > 0:
                            small_row = small.row(0, named=True)
                            steps_small = np.array(small_row["step"])
                            svs_small_arr = np.array(small_row[sv_col])

                        for sv_idx in range(n_plot_svs):
                            color = sv_cmap(sv_idx)

                            # Raw SVs
                            if svs_large_arr is not None:
                                ax_raw.plot(
                                    steps_large, svs_large_arr[:, sv_idx],
                                    color=color, ls="--", linewidth=1.0, alpha=0.7,
                                )
                            if svs_small_arr is not None:
                                label = (
                                    f"$\\sigma_{sv_idx}$"
                                    if row == 0 and gi == len(gammas) - 1
                                    else None
                                )
                                ax_raw.plot(
                                    steps_small, svs_small_arr[:, sv_idx],
                                    color=color, linewidth=1.0, alpha=0.85,
                                    label=label,
                                )

                            # Ratio
                            if svs_large_arr is not None and svs_small_arr is not None:
                                common = np.intersect1d(steps_large, steps_small)
                                if len(common) > 0:
                                    idx_l = np.searchsorted(steps_large, common)
                                    idx_s = np.searchsorted(steps_small, common)
                                    sv_l = svs_large_arr[idx_l, sv_idx]
                                    sv_s = svs_small_arr[idx_s, sv_idx]
                                    with np.errstate(divide="ignore", invalid="ignore"):
                                        ratio = sv_l / sv_s
                                    ax_rat.plot(
                                        common, ratio,
                                        color=color, linewidth=1.0, alpha=0.85,
                                    )

                        ax_raw.set_ylim(bottom=0)
                        ax_rat.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

                        # Labels
                        if row == 0:
                            ax_raw.set_title(
                                f"{GAMMA_NAMES[gamma]} — SVs", fontsize=10,
                            )
                            ax_rat.set_title(
                                f"{GAMMA_NAMES[gamma]} — Ratio", fontsize=10,
                            )
                        if gi == 0:
                            ax_raw.set_ylabel(f"lr={lr:.1e}", fontsize=9)
                        if row == n_lrs - 1:
                            ax_raw.set_xlabel("Step", fontsize=9)
                            ax_rat.set_xlabel("Step", fontsize=9)

                # Legend from top-right raw subplot
                handles, labels = axes[0, -2].get_legend_handles_labels()
                if handles:
                    handles.append(Line2D([0], [0], color="gray", ls="--", lw=1.0))
                    labels.append(large_label)
                    handles.append(Line2D([0], [0], color="gray", ls="-", lw=1.0))
                    labels.append(small_label)
                    fig.legend(
                        handles, labels,
                        loc="center right",
                        bbox_to_anchor=(1.0, 0.5),
                        fontsize=8,
                    )

                fig.suptitle(
                    f"{large_label} (dashed) vs {small_label} (solid)"
                    f" — {_pp_label(pp_i, pp_j)} | {mode_label}",
                    fontsize=13,
                    fontweight="bold",
                )
                fig.tight_layout(rect=[0, 0, 0.93, 0.96])

                out_path = (
                    FIGURES_PATH
                    / f"sv_compare_pp{pp_i}{pp_j}_bs{small_bs}_{mode_label.lower()}.png"
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {out_path}")

    print(f"\nAll plots saved to {FIGURES_PATH}/")


if __name__ == "__main__":
    main()
