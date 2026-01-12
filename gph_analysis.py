"""
GPH Experiment Analysis (Offline Only)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from runner import load_run
from plotting import compute_ci
from dln.results import RunResult


# %%
# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("outputs/gph")
FIGURES_PATH = Path("figures/gph")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [1, 2, 5, 10, 50]
N_SEEDS = 101

WIDTHS = [10, 50, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}
METRICS = ["grad_norm_squared", "trace_gradient_covariance", "trace_hessian_covariance"]
METRIC_LABELS = {
    "grad_norm_squared": "||∇L||²",
    "trace_gradient_covariance": "Tr(Σ)",
    "trace_hessian_covariance": "Tr(HΣ)",
}


# %%
# =============================================================================
# Data Loading
# =============================================================================


def get_path(
    width: int, gamma: float, batch: int | None, seed: int, noise: float
) -> Path:
    b_str = "None" if batch is None else str(batch)
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s{seed}_onlineFalse_noise{noise}"


def load_gd(width: int, gamma: float, noise: float) -> RunResult | None:
    path = get_path(width, gamma, None, 0, noise)
    return load_run(path) if path.exists() else None


def load_sgd(
    width: int, gamma: float, noise: float, batch_size: int
) -> list[RunResult]:
    runs = []
    for seed in range(N_SEEDS):
        path = get_path(width, gamma, batch_size, seed, noise)
        if path.exists() and (path / "history.json").exists():
            runs.append(load_run(path))
    return runs


# %%
# =============================================================================
# Plot Helpers
# =============================================================================

SYMLOG_THRESH = 1e-8


def symlog(x: np.ndarray) -> np.ndarray:
    """Symmetric log: sign(x) * log10(|x|), with threshold for small values."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x) > SYMLOG_THRESH
    result[mask] = np.sign(x[mask]) * np.log10(np.abs(x[mask]))
    result[~mask] = x[~mask] / SYMLOG_THRESH * np.log10(SYMLOG_THRESH)
    return result


def symlog_ticks(ax: plt.Axes) -> None:
    """Set readable tick labels for symmetric log scale."""
    ticks = [-1e4, -1e2, -1e0, -1e-2, 0, 1e-2, 1e0, 1e2, 1e4]
    positions = symlog(np.array(ticks))
    ax.set_yticks(positions)
    ax.set_yticklabels(["−10⁴", "−10²", "−1", "−10⁻²", "0", "10⁻²", "1", "10²", "10⁴"])


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# %%
# =============================================================================
# 1. Loss: SGD (averaged + CI) vs GD, shaded where GD < E[SGD]
# =============================================================================

for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            fig, axes = plt.subplots(1, len(WIDTHS), figsize=(6 * len(WIDTHS), 5))
            if len(WIDTHS) == 1:
                axes = [axes]

            for ax, width in zip(axes, WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    ax.set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])
                gd_loss = np.array(gd["train_loss"])
                sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
                sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_losses)

                ax.plot(steps, gd_loss, label="GD", color="C0")
                ax.plot(steps, sgd_mean, label=f"SGD (n={len(sgd_runs)})", color="C1")
                ax.fill_between(steps, sgd_lower, sgd_upper, alpha=0.3, color="C1")

                ax.fill_between(
                    steps,
                    0,
                    1,
                    where=(gd_loss < sgd_mean),
                    alpha=0.4,
                    color="green",
                    transform=ax.get_xaxis_transform(),
                )

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Train Loss")
                ax.set_title(f"Width={width}")
                ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"loss_g{gamma}_noise{noise}_b{batch_size}")

print("Loss plots saved.")


# %%
# =============================================================================
# 2. Loss with CI-based shading (GD < lower 95% CI of SGD)
# =============================================================================

for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            fig, axes = plt.subplots(1, len(WIDTHS), figsize=(6 * len(WIDTHS), 5))
            if len(WIDTHS) == 1:
                axes = [axes]

            for ax, width in zip(axes, WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    ax.set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])
                gd_loss = np.array(gd["train_loss"])
                sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
                sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_losses)

                ax.plot(steps, gd_loss, label="GD", color="C0")
                ax.plot(steps, sgd_mean, label=f"SGD (n={len(sgd_runs)})", color="C1")
                ax.fill_between(steps, sgd_lower, sgd_upper, alpha=0.3, color="C1")

                ax.fill_between(
                    steps,
                    0,
                    1,
                    where=(gd_loss < sgd_lower),
                    alpha=0.4,
                    color="green",
                    transform=ax.get_xaxis_transform(),
                )

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Train Loss")
                ax.set_title(f"Width={width}")
                ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size} — shaded: GD < CI lower"
            )
            save(fig, f"loss_ci_g{gamma}_noise{noise}_b{batch_size}")

print("Loss CI plots saved.")


# %%
# =============================================================================
# 3. Metrics (log scale, symlog for trace_hessian_covariance)
# =============================================================================

for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            gd = load_gd(WIDTHS[0], gamma, noise)
            if not gd or not gd.has("grad_norm_squared"):
                continue

            fig, axes = plt.subplots(
                len(METRICS), len(WIDTHS), figsize=(6 * len(WIDTHS), 4 * len(METRICS))
            )
            axes = np.atleast_2d(axes)

            for col, width in enumerate(WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    continue

                steps = np.array(gd["step"])

                for row, metric in enumerate(METRICS):
                    ax = axes[row, col]
                    use_symlog = metric == "trace_hessian_covariance"

                    gd_vals = np.array(gd[metric])
                    sgd_vals = [np.array(r[metric]) for r in sgd_runs]
                    sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_vals)

                    if use_symlog:
                        gd_vals = symlog(gd_vals)
                        sgd_mean, sgd_lower, sgd_upper = (
                            symlog(sgd_mean),
                            symlog(sgd_lower),
                            symlog(sgd_upper),
                        )

                    ax.plot(steps, gd_vals, label="GD", color="C0")
                    ax.plot(steps, sgd_mean, label="SGD", color="C1")
                    ax.fill_between(steps, sgd_lower, sgd_upper, alpha=0.3, color="C1")

                    if use_symlog:
                        symlog_ticks(ax)
                    else:
                        ax.set_yscale("log")

                    ax.set_xlabel("Step")
                    ax.set_ylabel(METRIC_LABELS[metric])
                    if row == 0:
                        ax.set_title(f"Width={width}")
                    if col == 0 and row == 0:
                        ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"metrics_g{gamma}_noise{noise}_b{batch_size}")

print("Metric plots saved.")


# %%
# =============================================================================
# 4. Signal-to-Noise Ratio: ||∇L||² / Tr(Σ)
# =============================================================================

for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            gd = load_gd(WIDTHS[0], gamma, noise)
            if not gd or not gd.has("grad_norm_squared"):
                continue

            fig, axes = plt.subplots(1, len(WIDTHS), figsize=(6 * len(WIDTHS), 5))
            if len(WIDTHS) == 1:
                axes = [axes]

            for ax, width in zip(axes, WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    ax.set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])

                # GD ratio (no noise covariance, so just plot grad norm)
                gd_grad = np.array(gd["grad_norm_squared"])
                gd_trace = np.array(gd["trace_gradient_covariance"])
                # Avoid division by zero
                gd_ratio = np.where(gd_trace > 1e-12, gd_grad / gd_trace, np.nan)

                # SGD: compute ratio for each run, then CI
                sgd_ratios = []
                for r in sgd_runs:
                    grad = np.array(r["grad_norm_squared"])
                    trace = np.array(r["trace_gradient_covariance"])
                    ratio = np.where(trace > 1e-12, grad / trace, np.nan)
                    sgd_ratios.append(ratio)

                sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_ratios)

                ax.plot(steps, gd_ratio, label="GD", color="C0")
                ax.plot(steps, sgd_mean, label=f"SGD (n={len(sgd_runs)})", color="C1")
                ax.fill_between(steps, sgd_lower, sgd_upper, alpha=0.3, color="C1")

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("||∇L||² / Tr(Σ)")
                ax.set_title(f"Width={width}")
                ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"snr_g{gamma}_noise{noise}_b{batch_size}")

print("Signal-to-noise ratio plots saved.")


# %%
# =============================================================================
# 5. Train vs Test Loss (GD and averaged SGD)
# =============================================================================

for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            fig, axes = plt.subplots(1, len(WIDTHS), figsize=(6 * len(WIDTHS), 5))
            if len(WIDTHS) == 1:
                axes = [axes]

            for ax, width in zip(axes, WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    ax.set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])

                # GD
                ax.plot(
                    steps, gd["train_loss"], label="GD train", color="C0", linestyle="-"
                )
                if gd.has("test_loss"):
                    ax.plot(
                        steps,
                        gd["test_loss"],
                        label="GD test",
                        color="C0",
                        linestyle="--",
                    )

                # SGD averaged
                sgd_train = [np.array(r["train_loss"]) for r in sgd_runs]
                train_mean, train_lower, train_upper = compute_ci(sgd_train)
                ax.plot(steps, train_mean, label="SGD train", color="C1", linestyle="-")
                ax.fill_between(steps, train_lower, train_upper, alpha=0.2, color="C1")

                if sgd_runs[0].has("test_loss"):
                    sgd_test = [np.array(r["test_loss"]) for r in sgd_runs]
                    test_mean, test_lower, test_upper = compute_ci(sgd_test)
                    ax.plot(
                        steps, test_mean, label="SGD test", color="C1", linestyle="--"
                    )
                    ax.fill_between(
                        steps, test_lower, test_upper, alpha=0.2, color="C1"
                    )

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title(f"Width={width}")
                ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"train_test_g{gamma}_noise{noise}_b{batch_size}")

print("Train/test plots saved.")


# %%
# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("GPH Summary: % of steps where L_GD < E[L_SGD]")
print("=" * 70)

for noise in NOISE_LEVELS:
    for batch_size in BATCH_SIZES:
        print(f"\nNoise={noise}, Batch={batch_size}")
        print("-" * 60)
        for width in WIDTHS:
            row = f"Width={width:3d}: "
            for gamma in GAMMAS:
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    row += f"γ={gamma}: N/A   "
                else:
                    gd_loss = np.array(gd["train_loss"])
                    sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
                    sgd_mean, _, _ = compute_ci(sgd_losses)
                    pct = (gd_loss < sgd_mean).mean() * 100
                    row += f"γ={gamma}: {pct:5.1f}%  "
            print(row)

print(f"\nFigures saved to: {FIGURES_PATH}")
