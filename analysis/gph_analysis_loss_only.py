"""
GPH Sweep Analysis

Analyzes results from GPH experiment sweeps (offline or online) with:
- Fast parquet-based data loading
- Caching of computed statistics for fast plot iteration
- Two plot types: loss ratio with significance testing, loss variability with spread bands
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy import stats as scipy_stats

from dln.results_io import load_sweep


# =============================================================================
# Configuration
# =============================================================================

WIDTHS = [10, 50, 100]
GAMMAS = [0.75, 1.0, 1.5]
NOISE_LEVELS = [0.0, 0.2]
MODEL_SEEDS = [0, 1]
BATCH_SIZES = [1, 2, 5, 10, 50]

GAMMA_MAX_STEPS = {0.75: 5000, 1.0: 8000, 1.5: 26000}
GAMMA_NAMES = {0.75: "NTK", 1.0: "Mean-Field", 1.5: "Saddle-to-Saddle"}

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/gph_offline"),
        "cache_path": Path("cache/gph_offline.pkl"),
        "figures_path": Path("figures/gph_offline"),
        "baseline_subdir": "full_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "regime_label": "Offline (finite data)",
    },
    "online": {
        "base_path": Path("outputs/gph_online"),
        "cache_path": Path("cache/gph_online.pkl"),
        "figures_path": Path("figures/gph_online"),
        "baseline_subdir": "large_batch",
        "sgd_subdir": "mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
        "regime_label": "Online (infinite data)",
    },
}

# Set in main() for plotting functions
_exp = None


# =============================================================================
# Data Loading
# =============================================================================


def _load_baseline(
    df: pl.DataFrame, width: int, gamma: float, noise: float, model_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int] | None:
    """Offline (full_batch): returns (steps, loss, None, 1).
    Online (large_batch): returns (steps, mean, var, n).
    """
    rows = df.filter(
        (pl.col("model.gamma") == gamma)
        & (pl.col("model.hidden_dim") == width)
        & (pl.col("model.model_seed") == model_seed)
        & (pl.col("data.noise_std") == noise)
    )

    if len(rows) == 0:
        return None

    steps = np.array(rows["step"][0])

    if len(rows) == 1:
        loss = np.array(rows["test_loss"][0])
        return steps, loss, None, 1

    # Multiple runs: compute mean and variance
    curves = np.array(rows["test_loss"].to_list())
    n = len(curves)
    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    return steps, mean, var, n


@dataclass
class ConfigStats:
    """Statistics for a single configuration."""

    steps: np.ndarray
    baseline_loss: np.ndarray
    baseline_var: np.ndarray | None  # None for deterministic (offline) baseline
    baseline_n: int
    sgd_mean: np.ndarray
    sgd_lower: np.ndarray
    sgd_upper: np.ndarray
    sgd_min: np.ndarray
    sgd_max: np.ndarray
    sgd_var: np.ndarray
    sgd_log_mean: np.ndarray  # mean of log(loss)
    sgd_log_std: np.ndarray  # std of log(loss), for log-space spread bands
    n_runs: int


def _compute_sgd_stats(
    df: pl.DataFrame,
    width: int,
    gamma: float,
    noise: float,
    model_seed: int,
    batch_size: int,
    baseline_steps: np.ndarray,
    baseline_loss: np.ndarray,
    baseline_var: np.ndarray | None,
    baseline_n: int,
) -> ConfigStats | None:
    """Vectorized mean, variance, CI, and log-space statistics over batch seeds."""
    rows = df.filter(
        (pl.col("model.gamma") == gamma)
        & (pl.col("model.hidden_dim") == width)
        & (pl.col("model.model_seed") == model_seed)
        & (pl.col("data.noise_std") == noise)
        & (pl.col("training.batch_size") == batch_size)
    )

    if len(rows) == 0:
        return None

    curves = np.array(rows["test_loss"].to_list())
    n = len(curves)
    log_curves = np.log(np.maximum(curves, 1e-300))

    mean = curves.mean(axis=0)
    var = curves.var(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    log_mean = log_curves.mean(axis=0)
    log_var = log_curves.var(axis=0, ddof=1) if n > 1 else np.zeros_like(log_mean)
    log_std = np.sqrt(log_var)
    min_vals = curves.min(axis=0)
    max_vals = curves.max(axis=0)

    sem = np.sqrt(var / n)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0

    return ConfigStats(
        steps=baseline_steps,
        baseline_loss=baseline_loss,
        baseline_var=baseline_var,
        baseline_n=baseline_n,
        sgd_mean=mean,
        sgd_lower=mean - t_val * sem,
        sgd_upper=mean + t_val * sem,
        sgd_min=min_vals,
        sgd_max=max_vals,
        sgd_var=var,
        sgd_log_mean=log_mean,
        sgd_log_std=log_std,
        n_runs=n,
    )


def compute_all_stats(exp_config: dict) -> dict[tuple, ConfigStats]:
    base_path = exp_config["base_path"]

    print("Loading data...")
    baseline_df = load_sweep(base_path / exp_config["baseline_subdir"])
    sgd_df = load_sweep(base_path / exp_config["sgd_subdir"])
    print(f"  Baseline: {len(baseline_df)} runs, SGD: {len(sgd_df)} runs")

    # Phase 1: Compute baselines
    print("Computing baselines...")
    baselines = {}
    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for width in WIDTHS:
                for model_seed in MODEL_SEEDS:
                    result = _load_baseline(
                        baseline_df, width, gamma, noise, model_seed
                    )
                    if result is not None:
                        baselines[(width, gamma, noise, model_seed)] = result

    total_baselines = len(GAMMAS) * len(NOISE_LEVELS) * len(WIDTHS) * len(MODEL_SEEDS)
    print(f"  Loaded {len(baselines)}/{total_baselines} baselines")

    # Phase 2: Compute SGD stats
    total = (
        len(GAMMAS) * len(NOISE_LEVELS) * len(BATCH_SIZES) * len(WIDTHS) * len(MODEL_SEEDS)
    )
    print(f"Computing SGD statistics ({total} configs)...")
    stats = {}
    done = 0

    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for batch_size in BATCH_SIZES:
                for width in WIDTHS:
                    for model_seed in MODEL_SEEDS:
                        bl_key = (width, gamma, noise, model_seed)
                        if bl_key not in baselines:
                            done += 1
                            continue
                        bl_steps, bl_loss, bl_var, bl_n = baselines[bl_key]
                        config_key = (width, gamma, noise, model_seed, batch_size)
                        result = _compute_sgd_stats(
                            sgd_df,
                            width,
                            gamma,
                            noise,
                            model_seed,
                            batch_size,
                            bl_steps,
                            bl_loss,
                            bl_var,
                            bl_n,
                        )
                        if result is not None:
                            stats[config_key] = result
                        done += 1
                        print(
                            f"\r  SGD stats: {done}/{total} ({100 * done / total:.0f}%)",
                            end="",
                            flush=True,
                        )

    print(f"\n  Computed {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def save_cache(stats: dict[tuple, ConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    for key, cs in stats.items():
        cache_data[key] = {
            "steps": cs.steps,
            "baseline_loss": cs.baseline_loss,
            "baseline_var": cs.baseline_var,
            "baseline_n": cs.baseline_n,
            "sgd_mean": cs.sgd_mean,
            "sgd_lower": cs.sgd_lower,
            "sgd_upper": cs.sgd_upper,
            "sgd_min": cs.sgd_min,
            "sgd_max": cs.sgd_max,
            "sgd_var": cs.sgd_var,
            "sgd_log_mean": cs.sgd_log_mean,
            "sgd_log_std": cs.sgd_log_std,
            "n_runs": cs.n_runs,
        }
    with open(path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        return {key: ConfigStats(**d) for key, d in cache_data.items()}
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    exp_config: dict, force_recompute: bool = False
) -> dict[tuple, ConfigStats]:
    cache_path = exp_config["cache_path"]
    if not force_recompute:
        stats = load_cache(cache_path)
        if stats is not None:
            print(f"Loaded {len(stats)} configurations from cache")
            return stats

    stats = compute_all_stats(exp_config)
    save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting
# =============================================================================


def save_fig(fig: plt.Figure, name: str) -> None:
    figures_path = _exp["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_path / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _welch_t_crit(se_a: np.ndarray, n_a: int, se_b: np.ndarray, n_b: int) -> np.ndarray:
    """Welch-Satterthwaite t critical value (two-tailed 95%), numerically stable.

    se_a, se_b are variance/n (squared standard error of the mean).
    When both SEs underflow to zero, returns 1.96 (normal approximation,
    appropriate since both means are precisely known).
    """
    se_sum = se_a + se_b
    ws_numer = se_sum**2
    ws_denom = se_a**2 / max(n_a - 1, 1) + se_b**2 / max(n_b - 1, 1)
    # When ws_denom underflows to 0, SEs are negligible → large df (normal approx)
    nonzero = ws_denom > 0
    df = np.where(nonzero, ws_numer / np.where(nonzero, ws_denom, 1.0), 1e6)
    df = np.maximum(df, 1.0)
    return scipy_stats.t.ppf(0.975, df=df)


def _significance_masks(s: ConfigStats) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sig_95, mean_below).

    For deterministic baselines (offline), reduces to a one-sample test
    against the SGD CI. For stochastic baselines (online), uses Welch's
    two-sample t-test on the difference.
    """
    mean_below = s.baseline_loss < s.sgd_mean

    if s.baseline_var is None:
        # Deterministic baseline: one-sample test
        sig_95 = s.baseline_loss < s.sgd_lower
    else:
        # Welch's two-sample test
        diff = s.sgd_mean - s.baseline_loss
        se_bl = s.baseline_var / s.baseline_n
        se_sgd = s.sgd_var / s.n_runs
        se_diff = np.sqrt(se_bl + se_sgd)

        t_crit = _welch_t_crit(se_bl, s.baseline_n, se_sgd, s.n_runs)

        # Lower bound of 95% CI for (E[SGD] - E[baseline]) > 0
        sig_95 = diff > t_crit * se_diff

    return sig_95, mean_below


def _ratio_ci(s: ConfigStats) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For deterministic baselines: divides SGD CI bounds by the constant.
    For stochastic baselines: delta method for the ratio of two means.
    """
    ratio = s.sgd_mean / s.baseline_loss

    if s.baseline_var is None:
        # Constant denominator: CI transforms linearly
        return ratio, s.sgd_lower / s.baseline_loss, s.sgd_upper / s.baseline_loss

    # Delta method: Var(Y/X) ≈ (Y/X)² × (Var(Y)/(n_Y·Y²) + Var(X)/(n_X·X²))
    safe_sgd = np.maximum(s.sgd_mean, 1e-30)
    safe_bl = np.maximum(s.baseline_loss, 1e-30)
    rel_var = s.sgd_var / (s.n_runs * safe_sgd**2) + s.baseline_var / (
        s.baseline_n * safe_bl**2
    )
    se_ratio = ratio * np.sqrt(rel_var)

    se_bl = s.baseline_var / s.baseline_n
    se_sgd = s.sgd_var / s.n_runs
    t_crit = _welch_t_crit(se_bl, s.baseline_n, se_sgd, s.n_runs)

    return ratio, ratio - t_crit * se_ratio, ratio + t_crit * se_ratio


def plot_loss_ratio(
    stats: dict[tuple, ConfigStats], gamma: float, noise: float, batch_size: int
) -> plt.Figure:
    """Ratio with 95% CI and significance shading."""
    bl_label = _exp["baseline_label"]
    regime = _exp["regime_label"]
    baseline_bs = _exp["baseline_batch_size"]

    # Construct labels based on experiment type
    if baseline_bs is not None:
        # Online: both are SGD with different batch sizes
        ratio_ylabel = f"$L_{{B={batch_size}}} \\,/\\, L_{{B={baseline_bs}}}$"
    else:
        # Offline: GD vs SGD
        ratio_ylabel = f"$L_{{\\mathrm{{SGD}}}} \\,/\\, L_{{\\mathrm{{{bl_label}}}}}$"

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
                ax_ratio.set_title(f"width={width}, model_seed={model_seed}\n(no data)")
                ax_ratio.set_xlabel("Training step")
                ax_ratio.set_ylabel(ratio_ylabel)
                ax_loss.set_xlabel("Training step")
                ax_loss.set_ylabel("Test loss")
                continue

            s = stats[key]

            # Build per-config labels
            if baseline_bs is not None:
                bl_legend = (
                    f"batch_size={baseline_bs} mean ({s.baseline_n} batch seeds)"
                )
                sgd_legend = f"batch_size={batch_size} mean ({s.n_runs} batch seeds)"
                sig_label = (
                    f"batch_size={baseline_bs} < batch_size={batch_size} (p < 0.05)"
                )
                nosig_label = (
                    f"batch_size={baseline_bs} < batch_size={batch_size} (not sig.)"
                )
            else:
                bl_legend = bl_label
                sgd_legend = f"SGD mean ({s.n_runs} batch seeds)"
                sig_label = f"{bl_label} < SGD (p < 0.05)"
                nosig_label = f"{bl_label} < SGD (not sig.)"

            # === Ratio plot (top) ===
            ratio_mean, ratio_lower, ratio_upper = _ratio_ci(s)

            ax_ratio.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax_ratio.plot(
                s.steps,
                ratio_mean,
                label=f"Mean ± 95% CI ({s.n_runs} batch seeds)",
                color="C1",
                linewidth=1.5,
            )
            ax_ratio.fill_between(
                s.steps, ratio_lower, ratio_upper, alpha=0.3, color="C1"
            )
            ax_ratio.set_xlabel("Training step")
            ax_ratio.set_ylabel(ratio_ylabel)
            ax_ratio.set_title(f"width={width}, model_seed={model_seed}")
            ax_ratio.legend(loc="upper right", fontsize=7)

            # === Loss comparison (bottom) ===
            ax_loss.plot(
                s.steps, s.baseline_loss, label=bl_legend, color="C0", linewidth=1.5
            )
            ax_loss.plot(
                s.steps,
                s.sgd_mean,
                label=sgd_legend,
                color="C1",
                linewidth=1.5,
            )
            # Log-transformed CI of the arithmetic mean (delta method on log(X̄))
            # Var(log X̄) ≈ SEM²/μ², so CI = mean ×/÷ exp(t × SEM/mean)
            t_val = scipy_stats.t.ppf(0.975, df=max(s.n_runs - 1, 1))
            sem = np.sqrt(s.sgd_var / s.n_runs)
            relative_sem = sem / np.maximum(s.sgd_mean, 1e-30)
            ci_factor = np.exp(t_val * relative_sem)
            ax_loss.fill_between(
                s.steps,
                s.sgd_mean / ci_factor,
                s.sgd_mean * ci_factor,
                alpha=0.3,
                color="C1",
            )

            sig_95, mean_below = _significance_masks(s)

            # Dark green: statistically significant at 95%
            ax_loss.fill_between(
                s.steps,
                0,
                1,
                where=sig_95,
                alpha=0.4,
                color="darkgreen",
                transform=ax_loss.get_xaxis_transform(),
                label=sig_label,
            )
            # Light green: mean below but not significant
            ax_loss.fill_between(
                s.steps,
                0,
                1,
                where=mean_below & ~sig_95,
                alpha=0.25,
                color="lightgreen",
                transform=ax_loss.get_xaxis_transform(),
                label=nosig_label,
            )

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Training step")
            ax_loss.set_ylabel("Test loss")
            ax_loss.legend(loc="upper right", fontsize=6)

    if baseline_bs is not None:
        suptitle = (
            f"Test Loss: batch_size={batch_size} vs batch_size={baseline_bs} — {regime}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), σ_ε={noise}"
        )
    else:
        suptitle = (
            f"Test Loss: SGD vs GD — {regime}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), σ_ε={noise}, batch_size={batch_size}"
        )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_loss_variance(
    stats: dict[tuple, ConfigStats], gamma: float, noise: float, batch_size: int
) -> plt.Figure:
    """Log-space standard deviation and test loss with spread bands."""
    bl_label = _exp["baseline_label"]
    regime = _exp["regime_label"]
    baseline_bs = _exp["baseline_batch_size"]

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
                ax_cv.set_title(f"width={width}, model_seed={model_seed}\n(no data)")
                ax_cv.set_xlabel("Training step")
                ax_cv.set_ylabel(r"$\sigma(\log L)$")
                ax_loss.set_xlabel("Training step")
                ax_loss.set_ylabel("Test loss")
                continue

            s = stats[key]

            # Build per-config labels
            if baseline_bs is not None:
                bl_legend = (
                    f"batch_size={baseline_bs} mean ({s.baseline_n} batch seeds)"
                )
                sgd_legend = f"batch_size={batch_size} mean ({s.n_runs} batch seeds)"
                cv_legend = f"batch_size={batch_size} ({s.n_runs} batch seeds)"
            else:
                bl_legend = bl_label
                sgd_legend = f"SGD mean ({s.n_runs} batch seeds)"
                cv_legend = f"SGD ({s.n_runs} batch seeds)"

            ax_cv.plot(
                s.steps, s.sgd_log_std, color="C1", linewidth=1.5, label=cv_legend
            )
            ax_cv.set_xlabel("Training step")
            ax_cv.set_ylabel(r"$\sigma(\log L)$")
            ax_cv.set_title(f"width={width}, model_seed={model_seed}")
            ax_cv.legend(loc="upper right", fontsize=7)

            ax_loss.plot(
                s.steps, s.baseline_loss, label=bl_legend, color="C0", linewidth=1.5
            )
            ax_loss.plot(
                s.steps,
                s.sgd_mean,
                label=sgd_legend,
                color="C1",
                linewidth=1.5,
            )

            spread_factor = np.exp(s.sgd_log_std)
            spread_lower = s.sgd_mean / spread_factor
            spread_upper = s.sgd_mean * spread_factor
            ax_loss.fill_between(
                s.steps,
                spread_lower,
                spread_upper,
                alpha=0.3,
                color="C1",
                label=r"mean $\times/\div\,$ exp($\sigma_{\log}$)",
            )

            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Training step")
            ax_loss.set_ylabel("Test loss")
            ax_loss.legend(loc="upper right", fontsize=6)

    if baseline_bs is not None:
        suptitle = (
            f"Test Loss Variability: batch_size={batch_size} vs batch_size={baseline_bs} — {regime}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), σ_ε={noise}"
        )
    else:
        suptitle = (
            f"SGD Test Loss Variability — {regime}\n"
            f"γ={gamma} ({GAMMA_NAMES[gamma]}), σ_ε={noise}, batch_size={batch_size}"
        )
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def generate_all_plots(stats: dict[tuple, ConfigStats]) -> None:
    figures_path = _exp["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)

    total = len(GAMMAS) * len(NOISE_LEVELS) * len(BATCH_SIZES) * 2
    completed = 0

    print(f"Generating {total} plots...")

    for gamma in GAMMAS:
        for noise in NOISE_LEVELS:
            for batch_size in BATCH_SIZES:
                fig = plot_loss_ratio(stats, gamma, noise, batch_size)
                save_fig(fig, f"loss_ratio_g{gamma}_noise{noise}_b{batch_size}")
                completed += 1
                print(
                    f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

                fig = plot_loss_variance(stats, gamma, noise, batch_size)
                save_fig(fig, f"loss_variance_g{gamma}_noise{noise}_b{batch_size}")
                completed += 1
                print(
                    f"\r  Progress: {completed}/{total} ({100 * completed / total:.0f}%)",
                    end="",
                    flush=True,
                )

    print(f"\nAll plots saved to {figures_path}/")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze GPH sweep results")
    parser.add_argument(
        "experiment",
        choices=["offline", "online"],
        help="Which experiment to analyze",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    args = parser.parse_args()

    global _exp
    _exp = EXPERIMENTS[args.experiment]

    print(f"Experiment: {args.experiment}")
    print(f"Data: {_exp['base_path']}")

    stats = get_stats(_exp, force_recompute=args.recompute)
    generate_all_plots(stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
