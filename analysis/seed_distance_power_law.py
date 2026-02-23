"""
Seed Distance (Power-Law Teacher) Analysis

Plots overlaid curves from a comparative sweep where two GD models with
different model seeds are trained on a power-law teacher matrix.

Produces two figures with columns by gamma:
  - Distances: param_distance, cosine similarity, and per-layer distances
  - Norms: per-layer norms for both models (A and B)

Usage:
    python analysis/seed_distance_power_law.py
    python analysis/seed_distance_power_law.py --input outputs/seed_distance_power_law
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import GAMMA_NAMES


DEFAULT_INPUT = Path("outputs/seed_distance_power_law")
DEFAULT_OUTPUT = Path("figures/seed_distance_power_law")


def _extract(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract a metric column as (n_runs, n_steps) array."""
    return np.vstack(df[col].to_list())


def _detect_layers(df: pl.DataFrame, prefix: str, suffix: str = "") -> int:
    i = 0
    while f"{prefix}{i}{suffix}" in df.columns:
        i += 1
    return i


def _compute_cosine_sim(
    df: pl.DataFrame, norm_col_a: str = "weight_norm_a", norm_col_b: str = "weight_norm_b",
) -> np.ndarray | None:
    """Derive cosine similarity from weight norms and param_distance.

    Uses the polarization identity:
        cos(a,b) = (||a||² + ||b||² - ||a-b||²) / (2·||a||·||b||)
    """
    if not all(c in df.columns for c in ("param_distance", norm_col_a, norm_col_b)):
        return None
    norm_a = _extract(df, norm_col_a)
    norm_b = _extract(df, norm_col_b)
    dist = _extract(df, "param_distance")
    denom = 2.0 * norm_a * norm_b
    cosine = np.where(
        denom > 0, (norm_a ** 2 + norm_b ** 2 - dist ** 2) / denom, 0.0,
    )
    return np.clip(cosine, -1.0, 1.0)


def _compute_ftle(
    df: pl.DataFrame, steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute finite-time Lyapunov exponent: λ(t) = (1/t) · log(d(t)/d(0)).

    Returns (ftle, ftle_steps) excluding step 0 where the exponent is undefined.
    """
    if "param_distance" not in df.columns:
        return None
    dist = _extract(df, "param_distance")
    d0 = dist[:, 0:1]  # (n_runs, 1)
    if np.any(d0 == 0):
        return None
    # Skip step 0
    t = steps[1:]
    ratio = dist[:, 1:] / d0
    # Avoid log(0) — clamp to tiny positive value
    ratio = np.maximum(ratio, 1e-30)
    ftle = np.log(ratio) / t[np.newaxis, :]
    return ftle, t


def _gamma_title(gamma: float) -> str:
    name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    return f"{name} (γ={gamma})"


def plot_distances(
    groups: list[tuple[float, pl.DataFrame]], output_dir: Path,
) -> None:
    """Plot param_distance, cosine similarity, and per-layer distances.

    Layout: rows = metric types, columns = gamma values.
    First row is always loss curves for training-phase context.
    """
    n_cols = len(groups)
    sample_df = groups[0][1]
    sample_steps = np.array(sample_df["step"][0])
    n_layers = _detect_layers(sample_df, "layer_distance_")
    has_cosine = _compute_cosine_sim(sample_df) is not None
    has_ftle = _compute_ftle(sample_df, sample_steps) is not None
    has_loss = "test_loss_a" in sample_df.columns

    n_rows = int(has_loss) + 1 + int(has_cosine) + int(has_ftle) + int(n_layers > 0)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False,
    )

    for col_idx, (gamma, df) in enumerate(groups):
        steps = np.array(df["step"][0])
        row = 0

        # Loss curves
        if has_loss:
            ax = axes[row, col_idx]
            loss_a = _extract(df, "test_loss_a")
            loss_b = _extract(df, "test_loss_b")
            ax.plot(steps, loss_a[0], color="C0", linewidth=1.5, label="Model A (seed 0)")
            for j in range(len(loss_b)):
                ax.plot(steps, loss_b[j], color="C1", alpha=0.12, linewidth=0.4)
            ax.plot(steps, loss_b.mean(axis=0), color="black", linewidth=1.5, linestyle="--", label="Model B mean")
            ax.set_yscale("log")
            if col_idx == 0:
                ax.set_ylabel("Test loss")
            ax.set_title(_gamma_title(gamma))
            ax.legend(fontsize=7)
            row += 1

        # Param distance
        ax = axes[row, col_idx]
        curves = _extract(df, "param_distance")
        for j in range(len(curves)):
            ax.plot(steps, curves[j], color="C0", alpha=0.15, linewidth=0.5)
        ax.plot(steps, curves.mean(axis=0), color="black", linewidth=1.5, label="Mean")
        ax.set_yscale("log")
        if col_idx == 0:
            ax.set_ylabel(r"$\|\theta_A - \theta_B\|$")
        if not has_loss:
            ax.set_title(_gamma_title(gamma))
        ax.legend(fontsize=7)
        row += 1

        # Cosine similarity
        if has_cosine:
            ax = axes[row, col_idx]
            cosine = _compute_cosine_sim(df)
            for j in range(len(cosine)):
                ax.plot(steps, cosine[j], color="C2", alpha=0.15, linewidth=0.5)
            ax.plot(steps, cosine.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            if col_idx == 0:
                ax.set_ylabel(r"$\cos(\theta_A, \theta_B)$")
            ax.legend(fontsize=7)
            row += 1

        # Finite-time Lyapunov exponent
        if has_ftle:
            ax = axes[row, col_idx]
            ftle, ftle_steps = _compute_ftle(df, steps)
            for j in range(len(ftle)):
                ax.plot(ftle_steps, ftle[j], color="C3", alpha=0.1, linewidth=0.4)
            ax.plot(ftle_steps, ftle.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            if col_idx == 0:
                ax.set_ylabel(r"$\lambda(t)$")
            ax.legend(fontsize=7)
            row += 1

        # Per-layer distances
        if n_layers > 0:
            ax = axes[row, col_idx]
            for i in range(n_layers):
                curves = _extract(df, f"layer_distance_{i}")
                mean = curves.mean(axis=0)
                color = f"C{i}"
                for j in range(len(curves)):
                    ax.plot(steps, curves[j], color=color, alpha=0.08, linewidth=0.4)
                ax.plot(steps, mean, color=color, linewidth=1.5, label=f"L{i} mean")
            ax.set_yscale("log")
            if col_idx == 0:
                ax.set_ylabel(r"$\|W_i^A - W_i^B\|_F$")
            ax.legend(fontsize=6, ncols=2)

        axes[-1, col_idx].set_xlabel("Training step")

    n_total = sum(len(df) for _, df in groups)
    fig.suptitle(
        f"Seed distance ({n_total} seed pairs) | power-law teacher",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "distances.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved distances.png")


def plot_norms(
    groups: list[tuple[float, pl.DataFrame]], output_dir: Path,
) -> None:
    """Plot per-layer norms for both models, all runs overlaid.

    Layout: rows = layers, columns = gamma values.
    """
    n_cols = len(groups)
    sample_df = groups[0][1]
    n_layers_a = _detect_layers(sample_df, "layer_norm_", "_a")
    n_layers_b = _detect_layers(sample_df, "layer_norm_", "_b")
    n_layers = max(n_layers_a, n_layers_b)
    if n_layers == 0:
        print("  No layer norm columns found, skipping norms plot.")
        return

    fig, axes = plt.subplots(
        n_layers, n_cols, figsize=(6 * n_cols, 3 * n_layers), squeeze=False,
    )

    for col_idx, (gamma, df) in enumerate(groups):
        steps = np.array(df["step"][0])

        for i in range(n_layers):
            ax = axes[i, col_idx]

            # Model A norms (should be identical across runs — same seed)
            col_a = f"layer_norm_{i}_a"
            if col_a in df.columns:
                curves_a = _extract(df, col_a)
                ax.plot(steps, curves_a[0], color="C0", linewidth=1.5, label="Model A (seed 0)")

            # Model B norms (vary across seed pairs)
            col_b = f"layer_norm_{i}_b"
            if col_b in df.columns:
                curves_b = _extract(df, col_b)
                for j in range(len(curves_b)):
                    ax.plot(steps, curves_b[j], color="C1", alpha=0.12, linewidth=0.4)
                ax.plot(
                    steps, curves_b.mean(axis=0),
                    color="black", linewidth=1.5, linestyle="--", label="Model B mean",
                )

            if col_idx == 0:
                ax.set_ylabel(rf"$\|W_{i}\|_F$")
            if i == 0:
                ax.set_title(_gamma_title(gamma))
            ax.legend(fontsize=6)

        axes[-1, col_idx].set_xlabel("Training step")

    n_total = sum(len(df) for _, df in groups)
    fig.suptitle(
        f"Layer norms ({n_total} seed pairs) | power-law teacher",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved norms.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot seed-distance curves from power-law teacher comparative sweep",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    path = args.input / "results.parquet"
    print(f"Loading {path}...")
    df = pl.read_parquet(path)
    print(f"  {len(df)} runs loaded")

    gammas = sorted(df["model.gamma"].unique().to_list(), reverse=True)
    groups = []
    for gamma in gammas:
        gdf = df.filter(pl.col("model.gamma") == gamma)
        print(f"  γ={gamma}: {len(gdf)} runs")
        groups.append((gamma, gdf))

    args.output.mkdir(parents=True, exist_ok=True)
    plot_distances(groups, args.output)
    plot_norms(groups, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
