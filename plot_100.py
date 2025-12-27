"""
Plot GD vs averaged SGD for width experiments.
"""

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from runner import load_run

# %%
BASE_PATH = Path("outputs/gph_w100")
FIGURES_PATH = Path("figures/gph_w100")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

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


def get_gd_path(width: int, gamma: float, online: bool, noise: float) -> Path:
    b_str = "500" if online else "None"
    return BASE_PATH / f"h{width}_g{gamma}_b{b_str}_s0_online{online}_noise{noise}"


def get_avg_path(width: int, gamma: float, online: bool, noise: float) -> Path:
    return (
        BASE_PATH / f"h{width}_g{gamma}_b{BATCH_SIZE}_avg_online{online}_noise{noise}"
    )


def load_gd(width: int, gamma: float, online: bool, noise: float):
    path = get_gd_path(width, gamma, online, noise)
    return load_run(path) if path.exists() else None


def load_sgd_avg(width: int, gamma: float, online: bool, noise: float) -> dict | None:
    path = get_avg_path(width, gamma, online, noise)
    if not path.exists():
        return None
    with open(path / "history.json") as f:
        return json.load(f)


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
    sgd_avg = load_sgd_avg(width, gamma, online, noise)

    if not gd or not sgd_avg:
        return []

    steps = np.array(gd["step"])
    gd_loss = np.array(gd["train_loss"])
    sgd_mean = np.array(sgd_avg["train_loss_mean"])

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

    has_data = False

    for width in WIDTHS:
        gd = load_gd(width, gamma, online, noise)
        sgd_avg = load_sgd_avg(width, gamma, online, noise)

        if not gd or not sgd_avg:
            print(f"Missing data: w={width}, {name}")
            continue

        has_data = True
        n_runs = sgd_avg["n_runs"]
        print(f"Plotting: w={width}, {name} (n={n_runs})")

        steps = np.array(gd["step"])
        gd_loss = np.array(gd["train_loss"])
        sgd_mean = np.array(sgd_avg["train_loss_mean"])

        # Shade GPH regions
        regions = get_regions(width, gamma, online, noise)
        shade_regions(ax, regions, color=WIDTH_COLORS[width])

        # Plot GD and SGD
        ax.plot(steps, gd_loss, label=f"w={width} GD")
        ax.plot(steps, sgd_mean, label=f"w={width} SGD (n={n_runs})")

    if not has_data:
        plt.close(fig)
        continue

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.legend()
    ax.set_title(f"{title}\n(Shaded: GPH holds, blue=w10, green=w100)")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"gd_vs_sgd_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"\nDone. Figures saved to: {FIGURES_PATH}")

# %%
