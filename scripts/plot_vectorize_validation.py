"""
Vectorized Training Validation Plot

Compares CPU (192 workers) vs GPU (--vectorize=auto) sweep results.
Checks correctness (identical loss curves) for both full-batch and mini-batch.

Produces a 2×3 grid:
  Row 1: Full batch comparison
  Row 2: Mini batch comparison
  Columns: loss curves overlay | final-loss parity | max-diff histogram

Usage (from project root):
    python scripts/plot_vectorize_validation.py

Expects outputs in outputs/vectorize_test/{cpu,gpu}/{full_batch,mini_batch}/.
Saves figure to figures/vectorize_validation.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# =============================================================================
# Config
# =============================================================================

BASE_DIR = Path("outputs/vectorize_test")
OUTPUT_PATH = Path("figures/vectorize_validation.png")

SWEEP_CONFIGS = {
    "Full Batch": {
        "cpu_dir": BASE_DIR / "cpu" / "full_batch",
        "gpu_dir": BASE_DIR / "gpu" / "full_batch",
    },
    "Mini Batch": {
        "cpu_dir": BASE_DIR / "cpu" / "mini_batch",
        "gpu_dir": BASE_DIR / "gpu" / "mini_batch",
    },
}


# =============================================================================
# Load and align
# =============================================================================

def load_and_join(cpu_dir: Path, gpu_dir: Path) -> pl.DataFrame:
    """Load both results and join on all shared scalar (non-list) columns."""
    df_cpu = pl.read_parquet(cpu_dir / "results.parquet")
    df_gpu = pl.read_parquet(gpu_dir / "results.parquet")

    # Identify join keys: all scalar (non-list) columns present in both
    cpu_scalar = {
        name for name, dtype in df_cpu.schema.items()
        if not isinstance(dtype, pl.List)
    }
    gpu_scalar = {
        name for name, dtype in df_gpu.schema.items()
        if not isinstance(dtype, pl.List)
    }
    join_keys = sorted(cpu_scalar & gpu_scalar)

    joined = df_cpu.join(df_gpu, on=join_keys, suffix="_gpu")

    print(f"  CPU: {len(df_cpu)} rows, GPU: {len(df_gpu)} rows, matched: {len(joined)}")
    if len(joined) < len(df_cpu):
        print(f"  WARNING: {len(df_cpu) - len(joined)} CPU runs not matched!")
    if len(joined) < len(df_gpu):
        print(f"  WARNING: {len(df_gpu) - len(joined)} GPU runs not matched!")

    return joined


def compute_diffs(joined: pl.DataFrame) -> dict:
    """Compare test_loss curves between CPU and GPU."""
    cpu_curves = joined["test_loss"].to_list()
    gpu_curves = joined["test_loss_gpu"].to_list()

    max_abs_diffs = []
    mean_abs_diffs = []
    final_cpu = []
    final_gpu = []

    for c_curve, g_curve in zip(cpu_curves, gpu_curves):
        c = np.array(c_curve)
        g = np.array(g_curve)
        diffs = np.abs(c - g)
        max_abs_diffs.append(diffs.max())
        mean_abs_diffs.append(diffs.mean())
        final_cpu.append(c[-1])
        final_gpu.append(g[-1])

    return {
        "cpu_curves": cpu_curves,
        "gpu_curves": gpu_curves,
        "max_abs_diffs": np.array(max_abs_diffs),
        "mean_abs_diffs": np.array(mean_abs_diffs),
        "final_cpu": np.array(final_cpu),
        "final_gpu": np.array(final_gpu),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_row(axes, label: str, diffs: dict) -> None:
    """Plot one row (3 panels) for a sweep comparison."""
    cmap = plt.cm.tab10

    # --- Panel 1: Overlaid loss curves ---
    ax = axes[0]
    n_show = min(20, len(diffs["cpu_curves"]))

    for i in range(n_show):
        color = cmap(i % 10)
        c = np.array(diffs["cpu_curves"][i])
        g = np.array(diffs["gpu_curves"][i])
        steps = np.arange(len(c))

        ax.plot(steps, c, color=color, alpha=0.6, linewidth=1.2,
                label="CPU" if i == 0 else None)
        ax.plot(steps, g, color=color, alpha=0.6, linewidth=1.2,
                linestyle="--", label="GPU vectorized" if i == 0 else None)

    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Test loss")
    ax.set_title(f"{label}: loss curves (first {n_show})")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # --- Panel 2: Final-loss parity scatter ---
    ax = axes[1]
    ax.scatter(diffs["final_cpu"], diffs["final_gpu"],
               alpha=0.4, s=10, edgecolors="none")

    lo = min(diffs["final_cpu"].min(), diffs["final_gpu"].min())
    hi = max(diffs["final_cpu"].max(), diffs["final_gpu"].max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("CPU final loss")
    ax.set_ylabel("GPU final loss")
    ax.set_title(f"{label}: final-loss parity")

    # --- Panel 3: Max absolute difference histogram ---
    ax = axes[2]
    ax.hist(diffs["max_abs_diffs"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(diffs["max_abs_diffs"].mean(), color="red", linestyle="--",
               label=f"Mean: {diffs['max_abs_diffs'].mean():.2e}")
    ax.axvline(diffs["max_abs_diffs"].max(), color="orange", linestyle="--",
               label=f"Max: {diffs['max_abs_diffs'].max():.2e}")
    ax.set_xlabel("Max |CPU - GPU| per run")
    ax.set_ylabel("Count")
    ax.set_title(f"{label}: max absolute diff")
    ax.legend(fontsize=8)


def plot_validation(all_diffs: dict) -> None:
    n_rows = len(all_diffs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row_idx, (label, diffs) in enumerate(all_diffs.items()):
        plot_row(axes[row_idx], label, diffs)

    plt.suptitle(
        "Vectorized Training Validation: CPU (192 workers) vs GPU (--vectorize=auto)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {OUTPUT_PATH}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(label: str, diffs: dict) -> None:
    n = len(diffs["max_abs_diffs"])
    worst = diffs["max_abs_diffs"].max()
    mean_max = diffs["max_abs_diffs"].mean()
    mean_mean = diffs["mean_abs_diffs"].mean()

    threshold = 1e-4
    n_pass = (diffs["max_abs_diffs"] < threshold).sum()

    print(f"\n  {label}:")
    print(f"    Runs compared:        {n}")
    print(f"    Max abs diff (worst): {worst:.2e}")
    print(f"    Max abs diff (mean):  {mean_max:.2e}")
    print(f"    Mean abs diff (mean): {mean_mean:.2e}")
    print(f"    Runs within {threshold:.0e}:    {n_pass}/{n}")

    if worst < 1e-3:
        print(f"    RESULT: PASS")
    else:
        print(f"    RESULT: FAIL — significant differences detected!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    all_diffs = {}

    print("=" * 60)
    print("VECTORIZED TRAINING VALIDATION")
    print("=" * 60)

    for label, paths in SWEEP_CONFIGS.items():
        cpu_dir = paths["cpu_dir"]
        gpu_dir = paths["gpu_dir"]

        if not cpu_dir.exists() or not gpu_dir.exists():
            print(f"\n  Skipping {label}: directories not found")
            print(f"    CPU: {cpu_dir} {'exists' if cpu_dir.exists() else 'MISSING'}")
            print(f"    GPU: {gpu_dir} {'exists' if gpu_dir.exists() else 'MISSING'}")
            continue

        print(f"\nLoading {label}...")
        joined = load_and_join(cpu_dir, gpu_dir)
        diffs = compute_diffs(joined)
        all_diffs[label] = diffs
        print_summary(label, diffs)

    if not all_diffs:
        print("\nNo data found. Run the sweep first:")
        print("  bash scripts/test_vectorize.sh")
        raise SystemExit(1)

    print("\n" + "=" * 60)
    overall_worst = max(d["max_abs_diffs"].max() for d in all_diffs.values())
    if overall_worst < 1e-3:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 60)

    plot_validation(all_diffs)
