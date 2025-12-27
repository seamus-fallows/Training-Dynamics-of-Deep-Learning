"""
GPH Experiment Analysis
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from runner import load_run
from plotting import plot, subtract_baseline
from dln.results import RunResult


# %%
# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("outputs/gph")
FIGURES_PATH = Path("figures/gph")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

LR = 0.0001
BATCH_SIZE = 5
N_SEEDS = 20

WIDTHS = [10, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]

GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}
METRICS = [
    "train_loss",
    "grad_norm_squared",
    "trace_gradient_covariance",
    "trace_hessian_covariance",
]
METRIC_LABELS = {
    "train_loss": "Loss",
    "grad_norm_squared": "||∇L||²",
    "trace_gradient_covariance": "Tr(Σ)",
    "trace_hessian_covariance": "Tr(HΣ)",
}

# %%
# =============================================================================
# Data Loading
# =============================================================================


def get_path(
    width: int, gamma: float, batch: int | None, seed: int, online: bool, noise: float
) -> Path:
    b_str = "None" if batch is None else str(batch)
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s{seed}_online{online}_noise{noise}"


def load_gd(width: int, gamma: float, online: bool, noise: float) -> RunResult | None:
    batch = 500 if online else None
    path = get_path(width, gamma, batch, 0, online, noise)
    return load_run(path) if path.exists() else None


def load_sgd(width: int, gamma: float, online: bool, noise: float) -> list[RunResult]:
    runs = []
    for seed in range(N_SEEDS):
        path = get_path(width, gamma, BATCH_SIZE, seed, online, noise)
        if path.exists() and (path / "history.json").exists():
            runs.append(load_run(path))
    return runs


# %%
# =============================================================================
# GPH-Specific Computation
# =============================================================================


def compute_gph_bound(gd: RunResult, sgd_runs: list[RunResult]) -> list[np.ndarray]:
    """||∇L_GD||² - ||∇L_SGD||² + (β/2)·Tr(H·Σ) for each SGD run. GPH holds if ≥ 0."""
    beta = LR / BATCH_SIZE
    gd_grad = np.array(gd["grad_norm_squared"])
    return [
        gd_grad
        - np.array(r["grad_norm_squared"])
        + (beta / 2) * np.array(r["trace_hessian_covariance"])
        for r in sgd_runs
    ]


# %%
# =============================================================================
# Iteration Helpers
# =============================================================================


def iter_by_gamma():
    """Yield (online, noise, gamma, name, title)."""
    for online in [False, True]:
        for noise in NOISE_LEVELS:
            for gamma in GAMMAS:
                mode = "online" if online else "offline"
                name = f"{mode}_noise{noise}_g{gamma}"
                title = (
                    f"{mode.title()}, noise={noise} — γ={gamma} ({GAMMA_NAMES[gamma]})"
                )
                yield online, noise, gamma, name, title


def iter_by_width():
    """Yield (online, noise, width, name, title)."""
    for online in [False, True]:
        for noise in NOISE_LEVELS:
            for width in WIDTHS:
                mode = "online" if online else "offline"
                name = f"{mode}_noise{noise}_w{width}"
                title = f"{mode.title()}, noise={noise} — Width={width}"
                yield online, noise, width, name, title


# %%
# =============================================================================
# Plot Helpers
# =============================================================================
def signed_log(x):
    """sign(x) * log10(|x| + 1)"""
    return np.sign(x) * np.log10(np.abs(x) + 1)


def signed_log_ticks(ax):
    """Add readable tick labels for signed log scale."""
    ticks = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
    tick_positions = [signed_log(t) for t in ticks]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(t) for t in ticks])


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_hline(ax: plt.Axes, y: float = 0) -> None:
    ax.axhline(y, color="black", linestyle="--", alpha=0.5)


def has_metrics(gamma: float, online: bool, noise: float) -> bool:
    gd = load_gd(WIDTHS[0], gamma, online, noise)
    return gd is not None and gd.has("grad_norm_squared")


def plot_widths(
    ax: plt.Axes,
    gamma: float,
    online: bool,
    noise: float,
    metric: str | None = None,
    derive=None,
    ylabel: str | None = None,
    hline: float | None = None,
    **plot_kwargs,
) -> None:
    """Plot data for all widths on one axes.

    Use metric= for recorded data, derive= for computed quantities.
    """
    if hline is not None:
        add_hline(ax, hline)

    data, steps = {}, None
    for width in WIDTHS:
        gd, sgd = (
            load_gd(width, gamma, online, noise),
            load_sgd(width, gamma, online, noise),
        )

        if metric is not None:
            if gd:
                data[f"w={width} GD"] = gd
            if sgd:
                data[f"w={width} SGD"] = sgd
        elif derive is not None and gd and sgd:
            steps = np.array(gd["step"])
            data[f"w={width}"] = derive(gd, sgd)

    if data:
        plot(data, metric=metric, steps=steps, ylabel=ylabel, ax=ax, **plot_kwargs)


# %%
# =============================================================================
# 1. Loss + Metrics
# =============================================================================

for online, noise, gamma, name, title in iter_by_gamma():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, m in enumerate(METRICS):
        plot_widths(
            axes.flat[i],
            gamma,
            online,
            noise,
            metric=m,
            ylabel=METRIC_LABELS[m],
            log_scale=(m == "train_loss"),
        )
    fig.suptitle(title)
    save(fig, f"curves_{name}")

print("Loss + metrics plots saved.")

# %%
# =============================================================================
# 2. Loss Difference
# =============================================================================

for online, noise, gamma, name, title in iter_by_gamma():
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_widths(
        ax,
        gamma,
        online,
        noise,
        derive=subtract_baseline,
        ylabel="E[L_SGD] - L_GD",
        hline=0,
        log_scale=False,
    )
    ax.set_ylim(-10, 10)
    ax.set_title(f"{title} — (GPH holds if ≤ 0)")
    save(fig, f"loss_diff_{name}")

print("Loss difference plots saved.")

# %%
# =============================================================================
# 3. GPH Bound
# =============================================================================

for online, noise, gamma, name, title in iter_by_gamma():
    if not has_metrics(gamma, online, noise):
        continue
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_widths(
        ax,
        gamma,
        online,
        noise,
        derive=compute_gph_bound,
        ylabel="GPH Bound",
        hline=0,
        log_scale=False,
    )
    ax.set_title(f"{title} — (GPH holds if ≥ 0)")
    save(fig, f"gph_bound_{name}")

print("GPH bound plots saved.")

# %%
# =============================================================================
# 4. Regime Comparison
# =============================================================================


def gamma_label(g: float) -> str:
    return f"γ={g} ({GAMMA_NAMES[g]})"


for online, noise, width, name, title in iter_by_width():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: GD loss by regime
    gd_data = {
        gamma_label(g): gd for g in GAMMAS if (gd := load_gd(width, g, online, noise))
    }
    if gd_data:
        plot(gd_data, ax=axes[0], title="GD Loss")

    # Right: Loss difference by regime
    add_hline(axes[1])
    diff_data, steps = {}, None
    for g in GAMMAS:
        gd, sgd = load_gd(width, g, online, noise), load_sgd(width, g, online, noise)
        if gd and sgd:
            steps = np.array(gd["step"])
            diff_data[gamma_label(g)] = subtract_baseline(gd, sgd)
    if diff_data:
        plot(
            diff_data,
            steps=steps,
            ylabel="E[L_SGD] - L_GD",
            log_scale=False,
            ax=axes[1],
            title="Loss Difference",
        )

    fig.suptitle(title)
    save(fig, f"regime_{name}")

print("Regime comparison plots saved.")

# %%
# =============================================================================
# 5. Summary Table
# =============================================================================

print("\n" + "=" * 80)
print("GPH Summary: % of training where E[L_SGD] ≤ L_GD")
print("=" * 80)

for online in [False, True]:
    for noise in NOISE_LEVELS:
        mode = "Online" if online else "Offline"
        print(f"\n{mode}, noise={noise}")
        print("-" * 60)

        for width in WIDTHS:
            row = f"Width={width:3d}: "
            for gamma in GAMMAS:
                gd, sgd = (
                    load_gd(width, gamma, online, noise),
                    load_sgd(width, gamma, online, noise),
                )

                if not (gd and sgd):
                    row += f"γ={gamma}: N/A   "
                else:
                    mean = np.mean(subtract_baseline(gd, sgd), axis=0)
                    row += f"γ={gamma}: {(mean <= 0).mean() * 100:5.1f}%  "
            print(row)

print("\n" + "=" * 80)
print(f"Done. Figures saved to: {FIGURES_PATH}")
