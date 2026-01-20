"""
Analysis for 3-dataseed experiment
gamma=0.75, noise=0.2, batch_size=5, widths=[100, 50, 10], data_seeds=[0, 1, 2], model_seeds=[0, 1]
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from runner import load_run
from dln.plotting import compute_ci


# %%
# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("outputs/gph_3dataseed_noise")
FIGURES_PATH = Path("figures/gph_3dataseed_noise")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

WIDTHS = [100, 50, 10]
GAMMA = 0.75
NOISE = 0.2
BATCH_SIZE = 5
N_BATCH_SEEDS = 100
DATA_SEEDS = [0, 1, 2, 3]
MODEL_SEEDS = [0, 1, 2, 3]


# %%
# =============================================================================
# Data Loading
# =============================================================================


def get_gd_path(width: int, data_seed: int, model_seed: int) -> Path:
    return BASE_PATH / f"h{width}_g{GAMMA}_bNone_s0_d{data_seed}_m{model_seed}"


def get_sgd_path(width: int, batch_seed: int, data_seed: int, model_seed: int) -> Path:
    return (
        BASE_PATH
        / f"h{width}_g{GAMMA}_b{BATCH_SIZE}_s{batch_seed}_d{data_seed}_m{model_seed}"
    )


def load_gd(width: int, data_seed: int, model_seed: int):
    path = get_gd_path(width, data_seed, model_seed)
    if not path.exists():
        return None
    try:
        return load_run(path)
    except Exception as e:
        print(f"  Warning: Failed to load GD {path}: {e}")
        return None


def load_sgd(width: int, data_seed: int, model_seed: int):
    runs = []
    for batch_seed in range(N_BATCH_SEEDS):
        path = get_sgd_path(width, batch_seed, data_seed, model_seed)
        if path.exists() and (path / "history.json").exists():
            try:
                runs.append(load_run(path))
            except Exception as e:
                print(f"  Warning: Failed to load {path}: {e}")
    return runs


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# %%
# =============================================================================
# Loss with two-tone shading (per data seed and model seed)
# =============================================================================

n_rows = len(DATA_SEEDS) * len(MODEL_SEEDS)
fig, axes = plt.subplots(n_rows, len(WIDTHS), figsize=(5 * len(WIDTHS), 4 * n_rows))

for row_idx, (data_seed, model_seed) in enumerate(
    [(d, m) for d in DATA_SEEDS for m in MODEL_SEEDS]
):
    for col, width in enumerate(WIDTHS):
        ax = axes[row_idx, col]

        print(
            f"Loading width={width}, data_seed={data_seed}, model_seed={model_seed}..."
        )
        gd = load_gd(width, data_seed, model_seed)
        sgd_runs = load_sgd(width, data_seed, model_seed)
        print(f"  GD: {'found' if gd else 'missing'}, SGD runs: {len(sgd_runs)}")

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
        if col == 0:
            ax.set_ylabel(f"d={data_seed}, m={model_seed}\nTrain Loss")
        if row_idx == 0:
            ax.set_title(f"Width={width}")
        if row_idx == 0 and col == len(WIDTHS) - 1:
            ax.legend(loc="upper right", fontsize=7)

fig.suptitle(f"γ={GAMMA} (NTK), noise={NOISE}, batch={BATCH_SIZE}")
save(fig, "loss_shading_by_dataseed_modelseed")
print(f"\nSaved: {FIGURES_PATH}/loss_shading_by_dataseed_modelseed.png")


# %%
# =============================================================================
# Summary statistics
# =============================================================================

print("\n" + "=" * 70)
print(f"Summary: γ={GAMMA}, noise={NOISE}, batch={BATCH_SIZE}")
print("=" * 70)

for width in WIDTHS:
    print(f"\n{'=' * 30} Width={width} {'=' * 30}")

    for data_seed in DATA_SEEDS:
        for model_seed in MODEL_SEEDS:
            gd = load_gd(width, data_seed, model_seed)
            sgd_runs = load_sgd(width, data_seed, model_seed)

            if not gd or not sgd_runs:
                print(f"  d={data_seed}, m={model_seed}: No data")
                continue

            steps = np.array(gd["step"])
            gd_loss = np.array(gd["train_loss"])
            sgd_losses = [np.array(r["train_loss"]) for r in sgd_runs]
            sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_losses)

            pct_significant = (gd_loss < sgd_lower).mean() * 100
            pct_better = (gd_loss < sgd_mean).mean() * 100

            print(f"\n  d={data_seed}, m={model_seed} ({len(sgd_runs)} runs):")
            print(f"    GD < E[SGD]:     {pct_better:5.1f}% of steps")
            print(
                f"    GD < CI lower:   {pct_significant:5.1f}% of steps (significant)"
            )

# %%
