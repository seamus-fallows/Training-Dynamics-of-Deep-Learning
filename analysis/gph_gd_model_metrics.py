"""
GD Model Metrics Analysis

Plots layer norms, gram norms, balance diffs, balance ratio, and effective
weight norm from the GD-only model metrics sweep.

Usage (run from the project root):

    python analysis/gph_gd_model_metrics.py

Options:
    --input PATH     Override input directory (default: outputs/gph_comparative_metrics/gd_metrics)
    --output PATH    Override figures directory (default: figures/gph_gd_model_metrics)

Data comes from a single-model (non-comparative) sweep.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

DEFAULT_INPUT = Path("outputs/gph_comparative_metrics/gd_metrics")
DEFAULT_OUTPUT = Path("figures/gph_gd_model_metrics")


# =============================================================================
# Data Loading
# =============================================================================


def load_data(input_dir: Path) -> pl.DataFrame:
    path = input_dir / "results.parquet"
    print(f"Loading {path}...")
    df = pl.read_parquet(path)
    print(f"  {len(df)} runs loaded")
    return df


def extract_array(row: dict, col: str) -> np.ndarray:
    return np.array(row[col])


# =============================================================================
# Plotting
# =============================================================================


def _detect_num_layers(df: pl.DataFrame) -> int:
    """Detect number of layers from column names."""
    i = 0
    while f"layer_norm_{i}" in df.columns:
        i += 1
    return i


def _suptitle(title: str, gamma: float, noise: float) -> str:
    gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    return f"{title} | {gamma_name} (γ={gamma}) | noise={noise}"


def plot_per_layer_metric(
    df: pl.DataFrame,
    gamma: float,
    noise: float,
    widths: list,
    model_seeds: list,
    num_layers: int,
    metric_prefix: str,
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> plt.Figure:
    """Generic plotter for per-layer metrics (one line per layer)."""
    n_cols = len(model_seeds)
    n_rows = len(widths)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False,
    )

    for w_idx, width in enumerate(widths):
        for col, seed in enumerate(model_seeds):
            ax = axes[w_idx, col]

            subset = df.filter(
                (pl.col("model.gamma") == gamma)
                & (pl.col("data.noise_std") == noise)
                & (pl.col("model.hidden_dim") == width)
                & (pl.col("model.model_seed") == seed)
            )

            if len(subset) == 0:
                if w_idx == 0:
                    ax.set_title(f"Seed {seed}")
                if col == 0:
                    ax.set_ylabel(f"Width {width}\n{ylabel}")
                continue

            row = subset.row(0, named=True)
            steps = extract_array(row, "step")

            for i in range(num_layers):
                col_name = f"{metric_prefix}_{i}"
                if col_name in row:
                    values = extract_array(row, col_name)
                    ax.plot(steps, values, label=f"Layer {i}", linewidth=1.2)

            if log_scale:
                ax.set_yscale("log")
            if w_idx == 0:
                ax.set_title(f"Seed {seed}")
            if col == 0:
                ax.set_ylabel(f"Width {width}\n{ylabel}")
            if w_idx == n_rows - 1:
                ax.set_xlabel("Training step")
            ax.legend(fontsize=6, loc="best")

    fig.suptitle(_suptitle(title, gamma, noise), fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_balance_ratio(
    df: pl.DataFrame,
    gamma: float,
    noise: float,
    widths: list,
    model_seeds: list,
    num_layers: int,
) -> plt.Figure:
    """Plot r_l = balance_diff_i / (gram_norm_i + gram_norm_{i+1})."""
    n_pairs = num_layers - 1
    n_cols = len(model_seeds)
    n_rows = len(widths)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False,
    )

    for w_idx, width in enumerate(widths):
        for col, seed in enumerate(model_seeds):
            ax = axes[w_idx, col]

            subset = df.filter(
                (pl.col("model.gamma") == gamma)
                & (pl.col("data.noise_std") == noise)
                & (pl.col("model.hidden_dim") == width)
                & (pl.col("model.model_seed") == seed)
            )

            if len(subset) == 0:
                if w_idx == 0:
                    ax.set_title(f"Seed {seed}")
                if col == 0:
                    ax.set_ylabel(f"Width {width}\n$r_l$")
                continue

            row = subset.row(0, named=True)
            steps = extract_array(row, "step")

            for i in range(n_pairs):
                diff = extract_array(row, f"balance_diff_{i}")
                gram_l = extract_array(row, f"gram_norm_{i}")
                gram_r = extract_array(row, f"gram_norm_{i + 1}")
                r_l = diff / (gram_l + gram_r)
                ax.plot(steps, r_l, label=f"Pair ({i}, {i+1})", linewidth=1.2)

            if w_idx == 0:
                ax.set_title(f"Seed {seed}")
            if col == 0:
                ax.set_ylabel(f"Width {width}\n$r_l$")
            if w_idx == n_rows - 1:
                ax.set_xlabel("Training step")
            ax.legend(fontsize=6, loc="best")

    fig.suptitle(
        _suptitle("Balance Ratio", gamma, noise),
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_effective_weight_norm(
    df: pl.DataFrame,
    gamma: float,
    noise: float,
    widths: list,
    model_seeds: list,
) -> plt.Figure:
    n_cols = len(model_seeds)
    n_rows = len(widths)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False,
    )

    for w_idx, width in enumerate(widths):
        for col, seed in enumerate(model_seeds):
            ax = axes[w_idx, col]

            subset = df.filter(
                (pl.col("model.gamma") == gamma)
                & (pl.col("data.noise_std") == noise)
                & (pl.col("model.hidden_dim") == width)
                & (pl.col("model.model_seed") == seed)
            )

            if len(subset) == 0:
                if w_idx == 0:
                    ax.set_title(f"Seed {seed}")
                if col == 0:
                    ax.set_ylabel(f"Width {width}\n$\\|W_{{eff}}\\|_F$")
                continue

            row = subset.row(0, named=True)
            steps = extract_array(row, "step")
            values = extract_array(row, "effective_weight_norm")

            ax.plot(steps, values, color="C0", linewidth=1.5)
            if w_idx == 0:
                ax.set_title(f"Seed {seed}")
            if col == 0:
                ax.set_ylabel(f"Width {width}\n$\\|W_{{eff}}\\|_F$")
            if w_idx == n_rows - 1:
                ax.set_xlabel("Training step")

    fig.suptitle(
        _suptitle("Effective Weight Norm", gamma, noise),
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GD model metrics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_data(args.input)
    num_layers = _detect_num_layers(df)
    print(f"  Detected {num_layers} layers")

    widths = sorted(df["model.hidden_dim"].unique().to_list())
    gammas = sorted(df["model.gamma"].unique().to_list())
    noise_levels = sorted(df["data.noise_std"].unique().to_list())
    model_seeds = sorted(df["model.model_seed"].unique().to_list())

    total = len(gammas) * len(noise_levels)
    completed = 0

    print(f"Generating figures for {total} (gamma, noise) configs...")

    for gamma in gammas:
        for noise in noise_levels:
            tag = f"g{gamma}_noise{noise}"

            fig = plot_per_layer_metric(
                df, gamma, noise, widths, model_seeds, num_layers,
                "layer_norm", "$\\|W_i\\|_F$", "Layer Norms",
            )
            save_fig(fig, f"layer_norms_{tag}", args.output)

            fig = plot_per_layer_metric(
                df, gamma, noise, widths, model_seeds, num_layers,
                "gram_norm", "$\\|W_i W_i^T\\|_F$", "Gram Norms",
            )
            save_fig(fig, f"gram_norms_{tag}", args.output)

            fig = plot_per_layer_metric(
                df, gamma, noise, widths, model_seeds, num_layers - 1,
                "balance_diff", "$\\|G_l\\|_F$", "Balance Diffs",
            )
            save_fig(fig, f"balance_diffs_{tag}", args.output)

            fig = plot_balance_ratio(
                df, gamma, noise, widths, model_seeds, num_layers,
            )
            save_fig(fig, f"balance_ratio_{tag}", args.output)

            fig = plot_effective_weight_norm(
                df, gamma, noise, widths, model_seeds,
            )
            save_fig(fig, f"effective_weight_norm_{tag}", args.output)

            completed += 1
            print(
                f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                end="", flush=True,
            )

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
