"""
Plot results from the underparameterized sweep.

Three figures, each with 3x3 grid (gamma x hidden_dim):
- Figure 1: Underparameterized (h ≤ 5) - hidden_dim in {3, 4, 5}
- Figure 2: Overparameterized (h > 5) - hidden_dim in {6, 8, 10}
- Figure 3: Very overparameterized (h >> 5) - hidden_dim in {40, 160, 640}

Each panel shows all seeds as separate lines.
"""

from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

from dln.results_io import load_sweep


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


def load_all_results(sweep_dir: Path) -> dict[tuple, list[tuple[int, dict]]]:
    results = defaultdict(list)

    if not (sweep_dir / "results.parquet").exists():
        print(f"Warning: {sweep_dir / 'results.parquet'} does not exist")
        return results

    df = load_sweep(sweep_dir)

    # Identify list columns (metric curves) vs scalar columns (params)
    list_cols = [c for c in df.columns if df[c].dtype == df["step"].dtype]

    for row in df.iter_rows(named=True):
        gamma = row["model.gamma"]
        hidden_dim = row["model.hidden_dim"]
        model_seed = row["model.model_seed"]

        history = {col: row[col] for col in list_cols}

        key = (gamma, hidden_dim)
        results[key].append((model_seed, history))

    # Sort by model_seed within each group
    for key in results:
        results[key].sort(key=lambda x: x[0])

    return results


# =============================================================================
# Plotting
# =============================================================================


def plot_panel(ax, data_list: list[tuple[int, dict]], title: str):
    """Each seed is a separate line."""
    if not data_list:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    for model_seed, history in data_list:
        color = SEED_COLORS[model_seed % len(SEED_COLORS)]
        steps = history["step"]

        if "train_loss" in history:
            ax.plot(steps, history["train_loss"], color=color, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_figure(results: dict, hidden_dims: list[int], suptitle: str) -> plt.Figure:
    """Rows = gamma, columns = hidden_dim."""
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
