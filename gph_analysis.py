"""
GPH Experiment Analysis (Offline Only)
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

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
N_SEEDS = 100

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


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def clip_below(arr: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Clip values below threshold for log scale plotting."""
    return np.clip(arr, threshold, None)


def share_ylim(axes: list[plt.Axes]) -> None:
    """Set all axes to share the same y limits (union of all)."""
    y_mins, y_maxs = [], []
    for ax in axes:
        ylim = ax.get_ylim()
        y_mins.append(ylim[0])
        y_maxs.append(ylim[1])
    y_min, y_max = min(y_mins), max(y_maxs)
    for ax in axes:
        ax.set_ylim(y_min, y_max)


# %%
# =============================================================================
# 1. Loss: SGD (averaged + CI) vs GD, two-tone shading
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

                # Dark green: GD < CI_lower (statistically significant)
                ax.fill_between(
                    steps,
                    0,
                    1,
                    where=(gd_loss < sgd_lower),
                    alpha=0.5,
                    color="darkgreen",
                    edgecolor="none",
                    transform=ax.get_xaxis_transform(),
                    label="GD < CI lower",
                )

                # Light green: CI_lower <= GD < E[SGD] (GD better but not significant)
                ax.fill_between(
                    steps,
                    0,
                    1,
                    where=(gd_loss >= sgd_lower) & (gd_loss < sgd_mean),
                    alpha=0.3,
                    color="lightgreen",
                    edgecolor="none",
                    transform=ax.get_xaxis_transform(),
                    label="CI lower ≤ GD < E[SGD]",
                )

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Train Loss")
                ax.set_title(f"Width={width}")
                ax.legend(loc="upper right", fontsize=8)

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"loss_g{gamma}_noise{noise}_b{batch_size}")

print("Loss plots saved.")


# %%
# =============================================================================
# 2. Loss Ratio: SGD / GD (all batch sizes on same plot)
# =============================================================================

# Color map for batch sizes
BATCH_COLORS = {1: "C0", 2: "C1", 5: "C2", 10: "C3", 50: "C4"}

# First pass: compute y-limits per noise level
noise_ylims = {}
for noise in NOISE_LEVELS:
    y_min, y_max = float("inf"), float("-inf")
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            for width in WIDTHS:
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)
                if not gd or not sgd_runs:
                    continue
                gd_loss = np.array(gd["train_loss"])
                sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
                sgd_ratios = [sgd / gd_loss for sgd in sgd_losses]
                _, ratio_lower, ratio_upper = compute_ci(sgd_ratios)
                y_min = min(y_min, np.nanmin(ratio_lower))
                y_max = max(y_max, np.nanmax(ratio_upper))
    margin = (y_max - y_min) * 0.05
    noise_ylims[noise] = (y_min - margin, y_max + margin)

# Second pass: create plots (one per gamma/noise, all batch sizes together)
for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        fig, axes = plt.subplots(1, len(WIDTHS), figsize=(6 * len(WIDTHS), 5))
        if len(WIDTHS) == 1:
            axes = [axes]

        for ax, width in zip(axes, WIDTHS):
            gd = load_gd(width, gamma, noise)
            if not gd:
                ax.set_title(f"Width={width} (no data)")
                continue

            steps = np.array(gd["step"])
            gd_loss = np.array(gd["train_loss"])

            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

            for batch_size in BATCH_SIZES:
                sgd_runs = load_sgd(width, gamma, noise, batch_size)
                if not sgd_runs:
                    continue

                sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
                sgd_ratios = [sgd / gd_loss for sgd in sgd_losses]
                ratio_mean, ratio_lower, ratio_upper = compute_ci(sgd_ratios)

                color = BATCH_COLORS.get(batch_size, "C5")
                ax.plot(steps, ratio_mean, label=f"b={batch_size}", color=color)
                ax.fill_between(steps, ratio_lower, ratio_upper, alpha=0.2, color=color)

            ax.set_ylim(noise_ylims[noise])
            ax.ticklabel_format(style="plain", axis="y")
            ax.set_xlabel("Step")
            ax.set_ylabel("SGD Loss / GD Loss")
            ax.set_title(f"Width={width}")
            ax.legend()

        fig.suptitle(f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}")
        save(fig, f"loss_ratio_g{gamma}_noise{noise}")

print("Loss ratio plots saved.")


# %%
# =============================================================================
# 2b. Power Analysis: Where would more runs help?
# =============================================================================


def samples_needed(effect_size, power=0.8, alpha=0.05):
    """Approximate sample size for detecting effect with given power."""
    if effect_size <= 0:
        return np.inf
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return n


for noise in NOISE_LEVELS:
    for gamma in GAMMAS:
        for batch_size in BATCH_SIZES:
            fig, axes = plt.subplots(2, len(WIDTHS), figsize=(6 * len(WIDTHS), 8))
            if len(WIDTHS) == 1:
                axes = axes.reshape(2, 1)

            for col, width in enumerate(WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    axes[0, col].set_title(f"Width={width} (no data)")
                    axes[1, col].set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])
                gd_loss = np.array(gd["train_loss"])
                sgd_losses = np.array([np.array(r["train_loss"]) for r in sgd_runs])

                sgd_mean = sgd_losses.mean(axis=0)
                sgd_std = sgd_losses.std(axis=0, ddof=1)
                n_runs = len(sgd_runs)

                # Effect size: (E[SGD] - GD) / std(SGD)
                # Positive means SGD > GD (GD is better)
                effect_size = np.where(
                    sgd_std > 1e-12, (sgd_mean - gd_loss) / sgd_std, 0
                )

                # Samples needed for 80% power
                n_needed = np.array([samples_needed(d) for d in effect_size])
                n_needed = np.clip(n_needed, 1, 10000)  # Cap for plotting

                # Regions where more runs would help:
                # - GD < E[SGD] (effect_size > 0)
                # - Effect size is meaningful (> 0.2)
                # - But not yet significant (need more runs than we have)
                sgd_sem = sgd_std / np.sqrt(n_runs)
                t_val = stats.t.ppf(0.975, df=n_runs - 1)
                sgd_lower = sgd_mean - t_val * sgd_sem

                more_runs_useful = (
                    (gd_loss < sgd_mean) & (gd_loss >= sgd_lower) & (effect_size > 0.2)
                )

                # Top plot: Effect size
                ax_effect = axes[0, col]
                ax_effect.axhline(0, color="gray", linestyle="-", alpha=0.3)
                ax_effect.axhline(
                    0.2, color="green", linestyle="--", alpha=0.5, label="Small (0.2)"
                )
                ax_effect.axhline(
                    0.5, color="orange", linestyle="--", alpha=0.5, label="Medium (0.5)"
                )
                ax_effect.axhline(
                    0.8, color="red", linestyle="--", alpha=0.5, label="Large (0.8)"
                )
                ax_effect.plot(steps, effect_size, color="C0", label="Effect size")

                # Shade regions where more runs would help
                ax_effect.fill_between(
                    steps,
                    ax_effect.get_ylim()[0] if ax_effect.get_ylim()[0] < 0 else 0,
                    effect_size,
                    where=more_runs_useful,
                    alpha=0.3,
                    color="yellow",
                    edgecolor="none",
                    label="More runs useful",
                )

                ax_effect.set_xlabel("Step")
                ax_effect.set_ylabel("Effect size d = (E[SGD] - GD) / σ")
                ax_effect.set_title(f"Width={width}")
                ax_effect.legend(loc="upper right", fontsize=7)
                ax_effect.set_ylim(-0.5, max(2, np.nanmax(effect_size) * 1.1))

                # Bottom plot: Samples needed for 80% power
                ax_power = axes[1, col]
                ax_power.axhline(
                    n_runs,
                    color="red",
                    linestyle="-",
                    alpha=0.7,
                    label=f"Current n={n_runs}",
                )
                ax_power.plot(steps, n_needed, color="C0", label="n needed (80% power)")

                # Shade where we need more runs than we have
                ax_power.fill_between(
                    steps,
                    n_runs,
                    n_needed,
                    where=(n_needed > n_runs) & (effect_size > 0),
                    alpha=0.3,
                    color="yellow",
                    edgecolor="none",
                    label="Underpowered",
                )

                ax_power.set_yscale("log")
                ax_power.set_xlabel("Step")
                ax_power.set_ylabel("Samples needed")
                ax_power.set_title(f"Width={width}")
                ax_power.legend(loc="upper right", fontsize=7)
                ax_power.set_ylim(1, 10000)

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}\n"
                f"Top: Effect size | Bottom: Runs needed for 80% power"
            )
            save(fig, f"power_g{gamma}_noise{noise}_b{batch_size}")

print("Power analysis plots saved.")


# %%
# =============================================================================
# 3. SGD Spread (SD bands in log space)
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

                # Compute envelope
                sgd_losses = np.array([np.array(r["train_loss"]) for r in sgd_runs])
                sgd_mean = sgd_losses.mean(axis=0)
                sgd_min = sgd_losses.min(axis=0)
                sgd_max = sgd_losses.max(axis=0)

                # Plot envelope
                ax.fill_between(
                    steps,
                    sgd_min,
                    sgd_max,
                    alpha=0.3,
                    color="C1",
                    edgecolor="none",
                    label="SGD (min/max)",
                )
                ax.plot(steps, sgd_mean, color="C1", linewidth=1.5, label="SGD mean")

                # Plot GD on top
                ax.plot(steps, gd_loss, label="GD", color="C0", linewidth=2)

                ax.set_yscale("log")
                ax.set_xlabel("Step")
                ax.set_ylabel("Train Loss")
                ax.set_title(f"Width={width}")
                ax.legend()

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"loss_spread_g{gamma}_noise{noise}_b{batch_size}")

print("Envelope plots saved.")


# %%
# =============================================================================
# 4. Metrics (log scale, dynamic clip for trace_hessian_covariance)
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

                    gd_vals = np.array(gd[metric])
                    sgd_vals = [np.array(r[metric]) for r in sgd_runs]
                    sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_vals)

                    # Dynamic clip threshold based on end-of-training values
                    if metric == "trace_hessian_covariance":
                        final_vals = [gd_vals[-1], sgd_mean[-1], sgd_lower[-1]]
                        final_vals = [v for v in final_vals if v > 0]
                        if final_vals:
                            clip_threshold = (
                                min(final_vals) * 0.1
                            )  # 10% of smallest final value
                        else:
                            clip_threshold = 1e-10
                        gd_vals = clip_below(gd_vals, clip_threshold)
                        sgd_mean = clip_below(sgd_mean, clip_threshold)
                        sgd_lower = clip_below(sgd_lower, clip_threshold)
                        sgd_upper = clip_below(sgd_upper, clip_threshold)

                    ax.plot(steps, gd_vals, label="GD", color="C0")
                    ax.plot(steps, sgd_mean, label="SGD", color="C1")
                    ax.fill_between(steps, sgd_lower, sgd_upper, alpha=0.3, color="C1")

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
# 5. Signal-to-Noise Ratio: ||∇L||² / Tr(Σ) — shared y-axis
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

            valid_axes = []

            for ax, width in zip(axes, WIDTHS):
                gd = load_gd(width, gamma, noise)
                sgd_runs = load_sgd(width, gamma, noise, batch_size)

                if not gd or not sgd_runs:
                    ax.set_title(f"Width={width} (no data)")
                    continue

                steps = np.array(gd["step"])

                gd_grad = np.array(gd["grad_norm_squared"])
                gd_trace = np.array(gd["trace_gradient_covariance"])
                gd_ratio = np.where(gd_trace > 1e-12, gd_grad / gd_trace, np.nan)

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
                valid_axes.append(ax)

            # Share y-axis across all valid subplots
            if valid_axes:
                share_ylim(valid_axes)

            fig.suptitle(
                f"γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}"
            )
            save(fig, f"snr_g{gamma}_noise{noise}_b{batch_size}")

print("Signal-to-noise ratio plots saved.")


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

# %%
