"""
Plot results from the underparameterized sweep.

Three figures, each with 3x3 grid (gamma x hidden_dim):
- Figure 1: Underparameterized (h ≤ 5) - hidden_dim in {3, 4, 5}
- Figure 2: Overparameterized (h > 5) - hidden_dim in {6, 8, 10}
- Figure 3: Very overparameterized (h >> 5) - hidden_dim in {40, 160, 640}

Each panel shows all seeds as separate lines.
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from runner import load_run
from dln.results import RunResult


# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("outputs/underparameterized")
OUTPUT_PATH = Path("figures")

TEACHER_RANK = 5

GAMMAS = [0.75, 1.0, 1.5]
UNDERPARAM_DIMS = [3, 4, 5]
OVERPARAM_DIMS = [6, 8, 10]
VERY_OVERPARAM_DIMS = [40, 160, 640]

GAMMA_NAMES = {
    0.75: "NTK (γ=0.75)",
    1.0: "Mean-Field (γ=1.0)",
    1.5: "Saddle-to-Saddle (γ=1.5)",
}

SEED_COLORS = plt.cm.tab10.colors


# =============================================================================
# Data Loading
# =============================================================================


def parse_subdir_name(name: str) -> dict:
    """Parse parameter values from subdirectory name."""
    params = {}

    gamma_match = re.search(r"gamma([\d.]+)", name)
    if gamma_match:
        params["gamma"] = float(gamma_match.group(1))

    hidden_match = re.search(r"hidden_dim(\d+)", name)
    if hidden_match:
        params["hidden_dim"] = int(hidden_match.group(1))

    seed_match = re.search(r"model_seed(\d+)", name)
    if seed_match:
        params["model_seed"] = int(seed_match.group(1))

    return params


def load_all_results(base_path: Path) -> dict[tuple, list[tuple[int, RunResult]]]:
    """
    Load all results from the sweep directory.

    Returns:
        dict mapping (gamma, hidden_dim) -> list of (model_seed, RunResult) tuples
    """
    results = defaultdict(list)

    if not base_path.exists():
        print(f"Warning: {base_path} does not exist")
        return results

    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir() or not (subdir / "history.json").exists():
            continue

        params = parse_subdir_name(subdir.name)
        gamma = params.get("gamma")
        hidden_dim = params.get("hidden_dim")
        model_seed = params.get("model_seed")

        if gamma is None or hidden_dim is None or model_seed is None:
            continue

        result = load_run(subdir)
        key = (gamma, hidden_dim)
        results[key].append((model_seed, result))

    # Sort by model_seed within each group
    for key in results:
        results[key].sort(key=lambda x: x[0])

    return results


# =============================================================================
# Plotting
# =============================================================================


def plot_panel(ax, data_list: list[tuple[int, RunResult]], title: str):
    """Plot train loss for a single (gamma, hidden_dim) panel. Each seed is a separate line."""
    if not data_list:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    for model_seed, result in data_list:
        color = SEED_COLORS[model_seed % len(SEED_COLORS)]
        steps = result["step"]

        if result.has("train_loss"):
            ax.plot(steps, result["train_loss"], color=color, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_figure(results: dict, hidden_dims: list[int], suptitle: str) -> plt.Figure:
    """Create a 3x3 grid: rows = gamma, columns = hidden_dim."""
    fig, axes = plt.subplots(
        len(GAMMAS), len(hidden_dims), figsize=(4 * len(hidden_dims), 3.5 * len(GAMMAS))
    )

    for row, gamma in enumerate(GAMMAS):
        for col, hidden_dim in enumerate(hidden_dims):
            ax = axes[row, col]
            key = (gamma, hidden_dim)
            data_list = results.get(key, [])

            gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
            title = f"{gamma_name}, h={hidden_dim}"

            plot_panel(ax, data_list, title)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


def main():
    print(f"Loading results from {BASE_PATH}...")
    results = load_all_results(BASE_PATH)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} (gamma, hidden_dim) combinations")
    for key, data_list in sorted(results.items()):
        print(f"  γ={key[0]}, hidden_dim={key[1]}: {len(data_list)} seeds")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Figure 1: Underparameterized
    fig1 = plot_figure(
        results, UNDERPARAM_DIMS, f"hidden dimension ≤ rank(teacher)={TEACHER_RANK}"
    )
    output_file1 = OUTPUT_PATH / "underparameterized.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file1}")
    plt.close(fig1)

    # Figure 2: Overparameterized
    fig2 = plot_figure(
        results, OVERPARAM_DIMS, f"hidden dimension > rank(teacher)={TEACHER_RANK}"
    )
    output_file2 = OUTPUT_PATH / "overparameterized.png"
    fig2.savefig(output_file2, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file2}")
    plt.close(fig2)

    # Figure 3: Very overparameterized
    fig3 = plot_figure(
        results,
        VERY_OVERPARAM_DIMS,
        f"hidden dimension >> rank(teacher)={TEACHER_RANK}",
    )
    output_file3 = OUTPUT_PATH / "very_overparameterized.png"
    fig3.savefig(output_file3, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file3}")
    plt.close(fig3)

    print("Done!")


if __name__ == "__main__":
    main()
