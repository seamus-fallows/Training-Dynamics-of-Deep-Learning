"""
GPH Sweep Analysis

Analyzes results from the GPH experiment sweep with:
- Parallel data loading across CPU cores
- Caching of computed statistics for fast plot iteration
- Three plot types: loss comparison, loss ratio, min/max spread
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats


# =============================================================================
# Configuration
# =============================================================================

BASE_PATH = Path("outputs/gph_offline")
CACHE_PATH = Path("cache/gph_offline.pkl")
FIGURES_PATH = Path("figures/gph_offline")

WIDTHS = [10, 50, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]
MODEL_SEEDS = [0, 1]
BATCH_SIZES = [1, 2, 5, 10, 50]

GAMMA_MAX_STEPS = {0.75: 6000, 1.0: 9000, 1.5: 27000}
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

N_WORKERS = os.cpu_count()
MAX_BATCH_SEED = 10000


# =============================================================================
# Data Loading
# =============================================================================


def get_gd_dir_name(width: int, gamma: float, noise: float, model_seed: int) -> str:
    max_steps = GAMMA_MAX_STEPS[gamma]
    return f"noise_std{noise}_max_steps{max_steps}_gamma{gamma}_hidden_dim{width}_model_seed{model_seed}_batch_sizeNone"


def load_history(path: Path) -> dict | None:
    history_file = path / "history.json"
    if not history_file.exists():
        return None
    try:
        with open(history_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def find_sgd_dirs(
    width: int, gamma: float, noise: float, model_seed: int, batch_size: int
) -> list[Path]:
    """Find all SGD run directories by constructing paths directly."""
    max_steps = GAMMA_MAX_STEPS[gamma]

    dirs = []
    for batch_seed in range(MAX_BATCH_SEED):
        name = f"noise_std{noise}_max_steps{max_steps}_gamma{gamma}_hidden_dim{width}_model_seed{model_seed}_batch_seed{batch_seed}_batch_size{batch_size}"
        path = BASE_PATH / name
        if path.exists():
            dirs.append(path)

    return dirs


def compute_ci(
    arrays: list[np.ndarray], confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and confidence interval from list of arrays."""
    if not arrays:
        return np.array([]), np.array([]), np.array([])

    data = np.stack(arrays)
    n = len(arrays)
    mean = data.mean(axis=0)

    if n < 2:
        return mean, mean, mean

    sem = data.std(axis=0, ddof=1) / np.sqrt(n)
    t_val = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)

    lower = mean - t_val * sem
    upper = mean + t_val * sem

    return mean, lower, upper


@dataclass
class ConfigStats:
    """Statistics for a single configuration."""

    steps: np.ndarray
    gd_loss: np.ndarray
    sgd_mean: np.ndarray
    sgd_lower: np.ndarray
    sgd_upper: np.ndarray
    sgd_min: np.ndarray
    sgd_max: np.ndarray
    sgd_var: np.ndarray
    n_runs: int


def compute_stats_for_config(
    width: int, gamma: float, noise: float, model_seed: int, batch_size: int
) -> ConfigStats | None:
    """Compute statistics for a single (width, gamma, noise, model_seed, batch_size) configuration."""
    # Load GD baseline
    gd_dir = BASE_PATH / get_gd_dir_name(width, gamma, noise, model_seed)
    gd_hist = load_history(gd_dir)
    if gd_hist is None:
        return None

    # Find and load all SGD runs
    sgd_dirs = find_sgd_dirs(width, gamma, noise, model_seed, batch_size)
    if not sgd_dirs:
        return None

    sgd_losses = []
    for path in sgd_dirs:
        hist = load_history(path)
        if hist and "test_loss" in hist:
            sgd_losses.append(np.array(hist["test_loss"]))

    if not sgd_losses:
        return None

    # Compute statistics
    steps = np.array(gd_hist["step"])
    gd_loss = np.array(gd_hist["test_loss"])

    sgd_mean, sgd_lower, sgd_upper = compute_ci(sgd_losses)
    sgd_arr = np.stack(sgd_losses)
    sgd_min = sgd_arr.min(axis=0)
    sgd_max = sgd_arr.max(axis=0)
    sgd_var = sgd_arr.var(axis=0, ddof=1)

    return ConfigStats(
        steps=steps,
        gd_loss=gd_loss,
        sgd_mean=sgd_mean,
        sgd_lower=sgd_lower,
        sgd_upper=sgd_upper,
        sgd_min=sgd_min,
        sgd_max=sgd_max,
        sgd_var=sgd_var,
        n_runs=len(sgd_losses),
    )


def _compute_wrapper(args: tuple) -> tuple[tuple, ConfigStats | None]:
    """Wrapper for parallel execution."""
    width, gamma, noise, model_seed, batch_size = args
    result = compute_stats_for_config(width, gamma, noise, model_seed, batch_size)
    return args, result


def compute_all_stats(n_workers: int = N_WORKERS) -> dict[tuple, ConfigStats]:
    """Compute statistics for all configurations in parallel."""
    # Generate all configuration tuples
    configs = [
        (width, gamma, noise, model_seed, batch_size)
        for gamma in GAMMAS
        for noise in NOISE_LEVELS
        for batch_size in BATCH_SIZES
        for width in WIDTHS
        for model_seed in MODEL_SEEDS
    ]

    print(
        f"Computing statistics for {len(configs)} configurations using {n_workers} workers..."
    )

    stats = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_wrapper, cfg): cfg for cfg in configs}

        for future in as_completed(futures):
            config, result = future.result()
            if result is not None:
                stats[config] = result
            completed += 1
            print(
                f"\r  Progress: {completed}/{len(configs)} ({100 * completed / len(configs):.0f}%)",
                end="",
                flush=True,
            )

    print(f"\nComputed stats for {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def save_cache(stats: dict[tuple, ConfigStats], path: Path = CACHE_PATH) -> None:
    """Save computed statistics to cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert ConfigStats to dicts for pickling
    cache_data = {}
    for key, cs in stats.items():
        cache_data[key] = {
            "steps": cs.steps,
            "gd_loss": cs.gd_loss,
            "sgd_mean": cs.sgd_mean,
            "sgd_lower": cs.sgd_lower,
            "sgd_upper": cs.sgd_upper,
            "sgd_min": cs.sgd_min,
            "sgd_max": cs.sgd_max,
            "sgd_var": cs.sgd_var,
            "n_runs": cs.n_runs,
        }

    with open(path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Cache saved to {path}")


def load_cache(path: Path = CACHE_PATH) -> dict[tuple, ConfigStats] | None:
    """Load statistics from cache file."""
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)

        # Convert dicts back to ConfigStats
        stats = {}
        for key, d in cache_data.items():
            stats[key] = ConfigStats(**d)

        return stats
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    force_recompute: bool = False, n_workers: int = N_WORKERS
) -> dict[tuple, ConfigStats]:
    """Get statistics, using cache if available."""
    if not force_recompute:
        stats = load_cache()
        if stats is not None:
            print(f"Loaded {len(stats)} configurations from cache")
            return stats

    stats = compute_all_stats(n_workers=n_workers)
    save_cache(stats)
    return stats


# =============================================================================
# Plotting
# =============================================================================


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure and close it."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_PATH / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_ratio(
    stats: dict[tuple, ConfigStats], gamma: float, noise: float, batch_size: int
) -> plt.Figure:
    """
    Plot SGD/GD loss ratio and shaded loss comparison.
    3x4 grid: columns = widths, rows = [seed0 ratio, seed0 loss, seed1 ratio, seed1 loss].
    Each subplot has its own y-axis scale.
    """
    fig, axes = plt.subplots(
        len(MODEL_SEEDS) * 2,
        len(WIDTHS),
        figsize=(5 * len(WIDTHS), 3.5 * len(MODEL_SEEDS) * 2),
    )
    axes = np.atleast_2d(axes)

    for seed_idx, model_seed in enumerate(MODEL_SEEDS):
        row_ratio = seed_idx * 2
        row_loss = seed_idx * 2 + 1

        for col, width in enumerate(WIDTHS):
            ax_ratio = axes[row_ratio, col]
            ax_loss = axes[row_loss, col]
            key = (width, gamma, noise, model_seed, batch_size)

            if key not in stats:
                ax_ratio.set_title(f"w={width}, seed={model_seed}\n(no data)")
                ax_ratio.set_xlabel("Step")
                ax_ratio.set_ylabel("SGD / GD")
                ax_loss.set_xlabel("Step")
                ax_loss.set_ylabel("Test Loss")
                continue

            s = stats[key]

            # === Ratio plot (top) ===
            ratio_mean = s.sgd_mean / s.gd_loss
            ratio_lower = s.sgd_lower / s.gd_loss
            ratio_upper = s.sgd_upper / s.gd_loss

            ax_ratio.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax_ratio.plot(
                s.steps, ratio_mean, label=f"n={s.n_runs}", color="C1", linewidth=1.5
            )
            ax_ratio.fill_between(
                s.steps, ratio_lower, ratio_upper, alpha=0.3, color="C1"
            )

            ax_ratio.set_xlabel("Step")
            ax_ratio.set_ylabel("SGD / GD")
            ax_ratio.set_title(f"w={width}, seed={model_seed}")
            ax_ratio.legend(loc="upper right", fontsize=7)

            # === Shaded loss plot (bottom) ===
            ax_loss.plot(s.steps, s.gd_loss, label="GD", color="C0", linewidth=1.5)
            ax_loss.plot(
                s.steps,
                s.sgd_mean,
                label=f"SGD (n={s.n_runs})",
                color="C1",
                linewidth=1.5,
            )
            ax_loss.fill_between(
                s.steps, s.sgd_lower, s.sgd_upper, alpha=0.3, color="C1"
            )

            # Shading: dark green where GD < CI lower (statistically significant)
            ax_loss.fill_between(
                s.steps,
                0,
                1,
                where=(s.gd_loss < s.sgd_lower),
                alpha=0.4,
                color="darkgreen",
                transform=ax_loss.get_xaxis_transform(),
                label="GD < CI lower",
            )

            # Shading: light green where CI lower <= GD < E[SGD]
            ax_loss.fill_between(
                s.steps,
                0,
                1,
                where=(s.gd_loss >= s.sgd_lower) & (s.gd_loss < s.sgd_mean),
                alpha=0.25,
                color="lightgreen",
                transform=ax_loss.get_xaxis_transform(),
                label="CI lower ≤ GD < E[SGD]",
            )

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Step")
            ax_loss.set_ylabel("Test Loss")
            ax_loss.legend(loc="upper right", fontsize=6)

    fig.suptitle(
        f"Loss Ratio & Comparison: γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_loss_spread(
    stats: dict[tuple, ConfigStats], gamma: float, noise: float, batch_size: int
) -> plt.Figure:
    """
    Plot min/max envelope of SGD runs vs GD.
    3x2 grid: columns = widths, rows = model_seeds.
    """
    fig, axes = plt.subplots(
        len(MODEL_SEEDS), len(WIDTHS), figsize=(5 * len(WIDTHS), 4 * len(MODEL_SEEDS))
    )
    axes = np.atleast_2d(axes)

    for row, model_seed in enumerate(MODEL_SEEDS):
        for col, width in enumerate(WIDTHS):
            ax = axes[row, col]
            key = (width, gamma, noise, model_seed, batch_size)

            if key not in stats:
                ax.set_title(f"w={width}, seed={model_seed}\n(no data)")
                ax.set_xlabel("Step")
                ax.set_ylabel("Test Loss")
                continue

            s = stats[key]

            # Plot envelope
            ax.fill_between(
                s.steps,
                s.sgd_min,
                s.sgd_max,
                alpha=0.3,
                color="C1",
                label=f"SGD min/max (n={s.n_runs})",
            )
            ax.plot(s.steps, s.sgd_mean, color="C1", linewidth=1.5, label="SGD mean")
            ax.plot(s.steps, s.gd_loss, color="C0", linewidth=2, label="GD")

            ax.set_yscale("log")
            ax.set_xlabel("Step")
            ax.set_ylabel("Test Loss")
            ax.set_title(f"w={width}, seed={model_seed}")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Loss Spread: γ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_loss_variance(
    stats: dict[tuple, ConfigStats], gamma: float, noise: float, batch_size: int
) -> plt.Figure:
    """
    Plot coefficient of variation (CV = std/mean) and loss with variance bands.
    3x4 grid: columns = widths, rows = [seed0 CV, seed0 loss±std, seed1 CV, seed1 loss±std].
    """
    fig, axes = plt.subplots(
        len(MODEL_SEEDS) * 2,
        len(WIDTHS),
        figsize=(5 * len(WIDTHS), 3.5 * len(MODEL_SEEDS) * 2),
    )
    axes = np.atleast_2d(axes)

    for seed_idx, model_seed in enumerate(MODEL_SEEDS):
        row_cv = seed_idx * 2
        row_loss = seed_idx * 2 + 1

        for col, width in enumerate(WIDTHS):
            ax_cv = axes[row_cv, col]
            ax_loss = axes[row_loss, col]
            key = (width, gamma, noise, model_seed, batch_size)

            if key not in stats:
                ax_cv.set_title(f"w={width}, seed={model_seed}\n(no data)")
                ax_cv.set_xlabel("Step")
                ax_cv.set_ylabel("CV")
                ax_loss.set_xlabel("Step")
                ax_loss.set_ylabel("Test Loss")
                continue

            s = stats[key]

            # Compute std and CV
            sgd_std = np.sqrt(s.sgd_var)
            cv = np.where(s.sgd_mean > 1e-12, sgd_std / s.sgd_mean, 0)

            # === CV plot (top) ===
            ax_cv.plot(s.steps, cv, color="C1", linewidth=1.5, label=f"n={s.n_runs}")
            ax_cv.set_xlabel("Step")
            ax_cv.set_ylabel("Std / Mean")
            ax_cv.set_title(f"w={width}, seed={model_seed}")
            ax_cv.legend(loc="upper right", fontsize=7)

            # === Loss with variance bands (bottom) ===
            ax_loss.plot(s.steps, s.gd_loss, label="GD", color="C0", linewidth=1.5)
            ax_loss.plot(
                s.steps,
                s.sgd_mean,
                label=f"SGD (n={s.n_runs})",
                color="C1",
                linewidth=1.5,
            )

            # Variance band: mean ± std
            sgd_lower_std = np.maximum(
                s.sgd_mean - sgd_std, 1e-12
            )  # Clip for log scale
            sgd_upper_std = s.sgd_mean + sgd_std
            ax_loss.fill_between(
                s.steps,
                sgd_lower_std,
                sgd_upper_std,
                alpha=0.3,
                color="C1",
                label="±1 std",
            )

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Step")
            ax_loss.set_ylabel("Test Loss")
            ax_loss.legend(loc="upper right", fontsize=6)

    fig.suptitle(
        f"SGD Loss: Relative Std & Mean±Std\nγ={gamma} ({GAMMA_NAMES[gamma]}), noise={noise}, batch={batch_size}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def generate_all_plots(stats: dict[tuple, ConfigStats]) -> None:
    """Generate all plots and save to FIGURES_PATH."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    total = len(GAMMAS) * len(NOISE_LEVELS) * len(BATCH_SIZES) * 3  # 3 plot types
    completed = 0

    print(f"Generating {total} plots...")

    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for batch_size in BATCH_SIZES:
                # Loss ratio (combined with shaded loss comparison)
                fig = plot_loss_ratio(stats, gamma, noise, batch_size)
                save_fig(fig, f"loss_ratio_g{gamma}_noise{noise}_b{batch_size}")
                completed += 1
                print(
                    f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

                # Loss spread
                fig = plot_loss_spread(stats, gamma, noise, batch_size)
                save_fig(fig, f"loss_spread_g{gamma}_noise{noise}_b{batch_size}")
                completed += 1
                print(
                    f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

                # Loss variance
                fig = plot_loss_variance(stats, gamma, noise, batch_size)
                save_fig(fig, f"loss_variance_g{gamma}_noise{noise}_b{batch_size}")
                completed += 1
                print(
                    f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

    print(f"\nAll plots saved to {FIGURES_PATH}/")


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary(stats: dict[tuple, ConfigStats]) -> None:
    """Print summary of where GD outperforms SGD."""
    print("\n" + "=" * 80)
    print("Summary: % of steps where GD < E[SGD]")
    print("=" * 80)

    for noise in NOISE_LEVELS:
        for batch_size in BATCH_SIZES:
            print(f"\nNoise={noise}, Batch={batch_size}")
            print("-" * 70)

            header = "           "
            for gamma in GAMMAS:
                header += f"  γ={gamma:4}  "
            print(header)

            for width in WIDTHS:
                for model_seed in MODEL_SEEDS:
                    row = f"w={width:3}, s={model_seed}: "
                    for gamma in GAMMAS:
                        key = (width, gamma, noise, model_seed, batch_size)
                        if key not in stats:
                            row += "   N/A   "
                        else:
                            s = stats[key]
                            pct = (s.gd_loss < s.sgd_mean).mean() * 100
                            row += f"  {pct:5.1f}%  "
                    print(row)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze GPH sweep results")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help=f"Number of parallel workers (default: {N_WORKERS})",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary statistics"
    )
    args = parser.parse_args()

    stats = get_stats(force_recompute=args.recompute, n_workers=args.workers)

    if args.summary:
        print_summary(stats)

    generate_all_plots(stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
