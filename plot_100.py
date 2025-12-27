"""
Plot GD vs averaged SGD for width experiments.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from runner import load_run
from plotting import plot, compute_ci

plt.style.use("seaborn-v0_8-whitegrid")

# %%
BASE_PATH = Path("outputs/gph_w100")
FIGURES_PATH = Path("figures/gph_w100")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

N_SEEDS = 100
BATCH_SIZE = 5

WIDTHS = [10, 100]
WIDTH_COLORS = {10: "blue", 100: "green"}
GAMMAS = [0.75, 1.0, 1.5]
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}
NOISE_LEVELS = [0.0, 0.2]
MIN_REGION_STEPS = 500


# %%
# =============================================================================
# Data Loading
# =============================================================================


def get_path(
    width: int, gamma: float, batch: int | None, seed: int, online: bool, noise: float
) -> Path:
    b_str = "None" if batch is None else str(batch)
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s{seed}_online{online}_noise{noise}"


def load_gd(width: int, gamma: float, online: bool, noise: float):
    batch = 500 if online else None
    path = get_path(width, gamma, batch, 0, online, noise)
    return load_run(path) if path.exists() else None


def load_sgd(width: int, gamma: float, online: bool, noise: float):
    runs = []
    for seed in range(N_SEEDS):
        path = get_path(width, gamma, BATCH_SIZE, seed, online, noise)
        if path.exists() and (path / "history.json").exists():
            runs.append(load_run(path))
    return runs


# %%
# =============================================================================
# Shading Helpers
# =============================================================================


def find_gph_holds_regions(gd_loss, sgd_mean, steps, min_steps):
    """Find contiguous regions where GD < SGD (GPH holds) for at least min_steps."""
    gph_holds = sgd_mean > gd_loss
    regions = []

    start_idx = None
    for i, holds in enumerate(gph_holds):
        if holds and start_idx is None:
            start_idx = i
        elif not holds and start_idx is not None:
            if steps[i - 1] - steps[start_idx] >= min_steps:
                regions.append((steps[start_idx], steps[i - 1]))
            start_idx = None

    if start_idx is not None and steps[-1] - steps[start_idx] >= min_steps:
        regions.append((steps[start_idx], steps[-1]))

    return regions


def shade_regions(ax, regions, color, alpha=0.15):
    for start, end in regions:
        ax.axvspan(start, end, color=color, alpha=alpha)


def get_regions(width: int, gamma: float, online: bool, noise: float):
    gd = load_gd(width, gamma, online, noise)
    sgd_runs = load_sgd(width, gamma, online, noise)

    if not gd or not sgd_runs:
        return []

    steps = np.array(gd["step"])
    gd_loss = np.array(gd["train_loss"])
    sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
    sgd_mean, _, _ = compute_ci(sgd_losses)

    return find_gph_holds_regions(gd_loss, sgd_mean, steps, MIN_REGION_STEPS)


# %%
# =============================================================================
# Iteration
# =============================================================================


def iter_configs():
    """Yield (online, noise, gamma, name, title)."""
    for online in [False, True]:
        for noise in NOISE_LEVELS:
            for gamma in GAMMAS:
                mode = "online" if online else "offline"
                name = f"{mode}_noise{noise}_g{gamma}"
                title = (
                    f"γ={gamma} ({GAMMA_NAMES[gamma]}) — {mode.title()}, noise={noise}"
                )
                yield online, noise, gamma, name, title


# %%
# =============================================================================
# GD vs SGD Loss
# =============================================================================

for online, noise, gamma, name, title in iter_configs():
    fig, ax = plt.subplots(figsize=(8, 5))

    data = {}
    has_data = False

    for width in WIDTHS:
        gd = load_gd(width, gamma, online, noise)
        sgd_runs = load_sgd(width, gamma, online, noise)

        if not gd or not sgd_runs:
            print(f"Missing data: w={width}, {name}")
            continue

        has_data = True
        print(f"Plotting: w={width}, {name} ({len(sgd_runs)} SGD runs)")

        regions = get_regions(width, gamma, online, noise)
        shade_regions(ax, regions, color=WIDTH_COLORS[width])

        data[f"w={width} GD"] = gd
        data[f"w={width} SGD (n={len(sgd_runs)})"] = sgd_runs

    if not has_data:
        plt.close(fig)
        continue

    plot(data, metric="train_loss", log_scale=True, ax=ax)
    ax.set_ylabel("Train Loss")
    ax.set_title(f"{title}\n(Shaded: GPH holds, blue=w10, green=w100)")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"gd_vs_sgd_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. Figures saved to: {FIGURES_PATH}")
