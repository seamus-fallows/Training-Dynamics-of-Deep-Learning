"""
Hessian Spectrum Analysis

Plots sorted eigenvalues λ_k vs index k for each (gamma, hidden_dim) combination.

Layout: rows = hidden_dim, columns = gamma (NTK, Mean-Field, Saddle-to-Saddle)

Usage (run from the project root):

    python analysis/hessian_spectrum.py offline
    python analysis/hessian_spectrum.py online
    python analysis/hessian_spectrum.py offline online

Options:
    --step STEP    Plot spectrum at a specific training step (default: final step)

Expects sweep outputs in outputs/hessian_spectrum/{offline,online}/results.parquet.
Saves figures to figures/hessian_spectrum/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import GAMMA_NAMES


GAMMAS = sorted(GAMMA_NAMES.keys())
DATA_DIR = Path("outputs/hessian_spectrum")
FIGURES_DIR = Path("figures/hessian_spectrum")

def load_data(regime: str) -> pl.DataFrame:
    path = DATA_DIR / regime / "results.parquet"
    return pl.read_parquet(path)


def plot_spectrum(
    df: pl.DataFrame,
    regime: str,
    step: int | None = None,
) -> plt.Figure:
    """One figure per regime: rows = hidden_dim, cols = gamma.

    Plots the sorted eigenvalue spectrum at the final evaluation step
    (or a specific step if given).
    """
    hidden_dims = sorted(df["model.hidden_dim"].unique().to_list())
    n_rows = len(hidden_dims)
    n_cols = len(GAMMAS)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False,
    )

    for col, gamma in enumerate(GAMMAS):
        axes[0, col].set_title(f"{GAMMA_NAMES[gamma]} (γ={gamma})", fontsize=11)

        for row, hdim in enumerate(hidden_dims):
            ax = axes[row, col]

            subset = df.filter(
                (pl.col("model.gamma") == gamma)
                & (pl.col("model.hidden_dim") == hdim)
            )

            if len(subset) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            batch_sizes = sorted(
                subset["training.batch_size"].unique().to_list(),
                key=lambda x: x if x is not None else float("inf"),
                reverse=True,
            )

            for ci, bs in enumerate(batch_sizes):
                if bs is None:
                    bs_subset = subset.filter(pl.col("training.batch_size").is_null())
                else:
                    bs_subset = subset.filter(pl.col("training.batch_size") == bs)
                label = "Full batch" if bs is None else f"B={bs}"
                color = f"C{ci}"

                for i in range(len(bs_subset)):
                    steps_list = bs_subset["step"][i].to_list()
                    spectra = bs_subset["hessian_spectrum"][i]

                    if step is not None:
                        idx = steps_list.index(step)
                    else:
                        idx = -1

                    eigenvalues = np.asarray(spectra[idx])
                    k = np.arange(1, len(eigenvalues) + 1)

                    ax.plot(
                        k, eigenvalues,
                        marker=".", markersize=2, linewidth=0.8,
                        color=color,
                        label=label if i == 0 else None,
                    )

            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
            linthresh = 1e-7 if hdim == 10 else 1e-10
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_xlabel("Index $k$")
            if col == 0:
                ax.set_ylabel(f"$\\lambda_k$  (width={hdim})")
            ax.legend(fontsize=7, loc="upper left")

    # Title
    regime_label = "Offline" if regime == "offline" else "Online"
    subtitle = f"step {step}" if step is not None else "final step"
    fig.suptitle(
        f"Hessian spectrum — {regime_label} ({subtitle})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot Hessian eigenvalue spectra")
    parser.add_argument(
        "regimes",
        nargs="+",
        choices=["offline", "online"],
        help="Which regime(s) to plot",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step to plot (default: final step)",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for regime in args.regimes:
        print(f"Loading {regime} data...")
        df = load_data(regime)
        print(f"  {len(df)} runs loaded")

        fig = plot_spectrum(df, regime, step=args.step)
        step_suffix = f"_step{args.step}" if args.step is not None else ""
        out_path = FIGURES_DIR / f"spectrum_{regime}{step_suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
