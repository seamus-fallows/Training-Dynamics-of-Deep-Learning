"""SV Dynamics Analysis — Top-k Power-Law Teacher

Averages SGD runs over batch seeds and compares against baseline (GD / large batch).

One figure per (partial_product, k, hidden_dim, model_seed, batch_size, regime).
Each figure: 3 rows × 3 columns
  - Row 1: Test loss (baseline + SGD mean ± 90% spread)
  - Row 2: Loss ratio (SGD / baseline, with 95% CI)
  - Row 3: Singular values (baseline solid, SGD mean dashed ± 90% spread)
  - Columns: γ = 0.75 (NTK), γ = 1.0 (Mean-Field), γ = 1.5 (Saddle-to-Saddle)

Usage:
    python analysis/sv_dynamics_topk.py offline
    python analysis/sv_dynamics_topk.py online
    python analysis/sv_dynamics_topk.py offline --recompute
    python analysis/sv_dynamics_topk.py offline --sort-parquet
"""

import argparse
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from _common import (
    CACHE_DIR, GAMMA_NAMES,
    build_filter, mean_centered_spread, sort_parquet,
)


# =============================================================================
# Configuration
# =============================================================================

L = 3 + 1  # num_hidden + 1
N_PLOT_CAP = 10
PARTIAL_PRODUCTS = [(i, j) for i in range(L) for j in range(i, L)]
PP_SV_COLS = [f"pp_{i}_{j}_sv" for i, j in PARTIAL_PRODUCTS]
GAMMAS = sorted(GAMMA_NAMES.keys())

# Key columns for this experiment (no noise_std, has k and max_steps zipped with gamma)
BL_KEY_COLS = [
    "data.params.k", "model.gamma", "max_steps",
    "model.hidden_dim", "model.model_seed",
]
BATCH_KEY_COLS = BL_KEY_COLS[:4]  # group reads by (k, gamma, max_steps, hidden_dim)

EXPERIMENTS = {
    "offline": {
        "base_path": Path("outputs/sv_dynamics_topk"),
        "cache_path": CACHE_DIR / "sv_dynamics_topk_offline.pkl",
        "figures_path": Path("figures/sv_dynamics_topk/offline"),
        "baseline_subdir": "offline/full_batch",
        "sgd_subdir": "offline/mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "sgd_batch_sizes": [1],
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
        "base_path": Path("outputs/sv_dynamics_topk"),
        "cache_path": CACHE_DIR / "sv_dynamics_topk_online.pkl",
        "figures_path": Path("figures/sv_dynamics_topk/online"),
        "baseline_subdir": "online/large_batch",
        "sgd_subdir": "online/mini_batch",
        "baseline_label": "Large batch",
        "baseline_batch_size": 500,
        "sgd_batch_sizes": [1],
        "regime_label": "Online (infinite data)",
    },
}


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SVStats:
    """Per-SV-column statistics: arrays of shape (n_steps, n_svs)."""
    mean: np.ndarray
    n: int
    spread_lo: np.ndarray | None  # None for n=1
    spread_hi: np.ndarray | None


@dataclass
class ConfigStats:
    """All plotting data for one (k, gamma, max_steps, hidden_dim, model_seed, batch_size)."""
    steps: np.ndarray
    n_runs: int
    # Loss
    baseline_loss: np.ndarray
    baseline_n: int
    baseline_spread_lo: np.ndarray | None
    baseline_spread_hi: np.ndarray | None
    sgd_loss_mean: np.ndarray
    sgd_loss_spread_lo: np.ndarray
    sgd_loss_spread_hi: np.ndarray
    # Loss ratio
    ratio: np.ndarray
    ratio_ci_lo: np.ndarray
    ratio_ci_hi: np.ndarray
    sig_mask: np.ndarray
    # SVs per partial product
    baseline_svs: dict[str, SVStats] = field(default_factory=dict)
    sgd_svs: dict[str, SVStats] = field(default_factory=dict)


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_sv_3d(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract SV data as (n_runs, n_steps, n_svs) array."""
    return np.stack([np.array(row) for row in df[col].to_list()])


def _make_sv_stats(curves_3d: np.ndarray) -> SVStats:
    """Compute mean and spread for a (n_runs, n_steps, n_svs) array."""
    n_runs, n_steps, n_svs = curves_3d.shape
    mean = curves_3d.mean(axis=0)

    if n_runs == 1:
        return SVStats(mean=mean, n=1, spread_lo=None, spread_hi=None)

    # Flatten (n_steps, n_svs) → (n_steps * n_svs) for mean_centered_spread
    flat = curves_3d.reshape(n_runs, -1)
    flat_mean = mean.reshape(-1)
    lo, hi = mean_centered_spread(flat, flat_mean)
    return SVStats(
        mean=mean, n=n_runs,
        spread_lo=lo.reshape(n_steps, n_svs),
        spread_hi=hi.reshape(n_steps, n_svs),
    )


def _welch_t_crit(
    se_a: np.ndarray, n_a: int,
    se_b: np.ndarray, n_b: int,
    quantile: float = 0.975,
) -> np.ndarray:
    """Welch-Satterthwaite t critical value (copied from gph_analysis_loss_only)."""
    se_sum = se_a + se_b
    ws_numer = se_sum**2
    ws_denom = se_a**2 / max(n_a - 1, 1) + se_b**2 / max(n_b - 1, 1)
    nonzero = ws_denom > 0
    df = np.where(nonzero, ws_numer / np.where(nonzero, ws_denom, 1.0), 1e6)
    df = np.maximum(df, 1.0)
    return scipy_stats.t.ppf(quantile, df=df)


@dataclass
class _BaselineStats:
    """Intermediate baseline data for computing ConfigStats."""
    steps: np.ndarray
    loss: np.ndarray
    loss_var: np.ndarray | None
    n: int
    spread_lo: np.ndarray | None
    spread_hi: np.ndarray | None
    svs: dict[str, SVStats]


def _compute_baseline(group_df: pl.DataFrame) -> _BaselineStats:
    steps = np.array(group_df["step"][0])
    loss_curves = np.vstack(group_df["test_loss"].to_list())
    n = len(loss_curves)

    loss = loss_curves.mean(axis=0) if n > 1 else loss_curves[0]
    loss_var = loss_curves.var(axis=0, ddof=1) if n > 1 else None

    if n > 1:
        spread_lo, spread_hi = mean_centered_spread(loss_curves, loss)
    else:
        spread_lo, spread_hi = None, None

    svs = {}
    for col in PP_SV_COLS:
        if col in group_df.columns:
            sv_3d = _extract_sv_3d(group_df, col)
            svs[col] = _make_sv_stats(sv_3d)
            del sv_3d

    return _BaselineStats(
        steps=steps, loss=loss, loss_var=loss_var, n=n,
        spread_lo=spread_lo, spread_hi=spread_hi, svs=svs,
    )


def _compute_config_stats(sgd_df: pl.DataFrame, bl: _BaselineStats) -> ConfigStats:
    loss_curves = np.vstack(sgd_df["test_loss"].to_list())
    n = len(loss_curves)
    mean = loss_curves.mean(axis=0)

    if n > 1:
        spread_lo, spread_hi = mean_centered_spread(loss_curves, mean)
        var = loss_curves.var(axis=0, ddof=1)
    else:
        spread_lo, spread_hi = mean, mean
        var = np.zeros_like(mean)

    # --- Loss ratio and CI (mirrors gph_analysis_loss_only.py) ---

    safe_bl = np.maximum(bl.loss, 1e-30)
    ratio = mean / safe_bl
    sem = np.sqrt(var / max(n, 1))

    df_sgd = max(n - 1, 1)

    if bl.loss_var is None:
        # Deterministic baseline — CI transforms linearly
        t_ci = scipy_stats.t.ppf(0.975, df=df_sgd)
        ratio_ci_lo = (mean - t_ci * sem) / safe_bl
        ratio_ci_hi = (mean + t_ci * sem) / safe_bl
        # One-sided: baseline < SGD at p<0.05
        t_sig = scipy_stats.t.ppf(0.95, df=df_sgd)
        sig_mask = bl.loss < (mean - t_sig * sem)
    else:
        # Stochastic baseline — delta method
        safe_sgd = np.maximum(mean, 1e-30)
        rel_var = var / (n * safe_sgd**2) + bl.loss_var / (bl.n * safe_bl**2)
        se_ratio = ratio * np.sqrt(np.maximum(rel_var, 0))

        se_bl = bl.loss_var / bl.n
        se_sgd = var / n
        t_crit = _welch_t_crit(se_bl, bl.n, se_sgd, n)
        ratio_ci_lo = ratio - t_crit * se_ratio
        ratio_ci_hi = ratio + t_crit * se_ratio

        # One-sided Welch's t-test
        t_sig = _welch_t_crit(se_bl, bl.n, se_sgd, n, quantile=0.95)
        diff = mean - bl.loss
        se_diff = np.sqrt(se_bl + se_sgd)
        sig_mask = diff > t_sig * se_diff

    # --- SV statistics ---

    sgd_svs = {}
    for col in PP_SV_COLS:
        if col in sgd_df.columns:
            sv_3d = _extract_sv_3d(sgd_df, col)
            sgd_svs[col] = _make_sv_stats(sv_3d)
            del sv_3d

    return ConfigStats(
        steps=bl.steps,
        n_runs=n,
        baseline_loss=bl.loss,
        baseline_n=bl.n,
        baseline_spread_lo=bl.spread_lo,
        baseline_spread_hi=bl.spread_hi,
        sgd_loss_mean=mean,
        sgd_loss_spread_lo=spread_lo,
        sgd_loss_spread_hi=spread_hi,
        ratio=ratio,
        ratio_ci_lo=ratio_ci_lo,
        ratio_ci_hi=ratio_ci_hi,
        sig_mask=sig_mask,
        baseline_svs=bl.svs,
        sgd_svs=sgd_svs,
    )


# =============================================================================
# Statistics Computation
# =============================================================================


def compute_all_stats(exp_config: dict) -> dict[tuple, ConfigStats]:
    baseline_dir = exp_config["base_path"] / exp_config["baseline_subdir"]
    sgd_dir = exp_config["base_path"] / exp_config["sgd_subdir"]

    bl_select = BL_KEY_COLS + ["step", "test_loss"] + PP_SV_COLS

    # Check if SGD parquet has a batch_size column (absent when sweep has single value)
    sgd_schema = pl.scan_parquet(sgd_dir / "results.parquet").collect_schema()
    has_bs_col = "training.batch_size" in sgd_schema.names()
    sgd_select = BL_KEY_COLS + (["training.batch_size"] if has_bs_col else []) + ["test_loss"] + PP_SV_COLS

    # Phase 1: Baselines
    print(f"Loading baselines from {baseline_dir}...")
    bl_df = (
        pl.scan_parquet(baseline_dir / "results.parquet")
        .select(bl_select)
        .collect()
    )
    bl_groups = bl_df.partition_by(BL_KEY_COLS, as_dict=True)

    print(f"Computing baselines ({len(bl_groups)} groups)...")
    baselines: dict[tuple, _BaselineStats] = {}
    for key, group_df in bl_groups.items():
        if len(group_df) > 0:
            baselines[key] = _compute_baseline(group_df)
    del bl_df, bl_groups
    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — batched by (k, gamma, max_steps, hidden_dim)
    sgd_lf = pl.scan_parquet(sgd_dir / "results.parquet")
    sgd_batch_sizes = exp_config["sgd_batch_sizes"]

    key_batches: dict[tuple, list[tuple]] = defaultdict(list)
    for bl_key in sorted(baselines):
        key_batches[bl_key[:4]].append(bl_key)

    n_batches = len(key_batches)
    total_configs = len(baselines)
    print(f"Computing SGD statistics ({n_batches} batches, {total_configs} baseline groups)...")

    stats: dict[tuple, ConfigStats] = {}
    completed = 0
    for batch_key, batch_bl_keys in key_batches.items():
        chunk = (
            sgd_lf.filter(build_filter(BATCH_KEY_COLS, batch_key))
            .select(sgd_select)
            .collect()
        )
        if has_bs_col:
            sub_groups = chunk.partition_by(
                ["model.model_seed", "training.batch_size"], as_dict=True,
            )
        else:
            # Single batch size — partition by model_seed only, pair with known batch_size
            sub_groups = {
                (ms, sgd_batch_sizes[0]): g
                for (ms,), g in chunk.partition_by(["model.model_seed"], as_dict=True).items()
            }
        del chunk

        for (model_seed, batch_size), group_df in sub_groups.items():
            bl_key = batch_key + (model_seed,)
            bl = baselines.get(bl_key)
            if bl is None or len(group_df) == 0:
                continue
            key = bl_key + (batch_size,)
            stats[key] = _compute_config_stats(group_df, bl)

        completed += len(batch_bl_keys)
        print(
            f"\r  SGD stats: {completed}/{total_configs}"
            f" ({100 * completed / total_configs:.0f}%)",
            end="", flush=True,
        )
        del sub_groups

    print(f"\n  Computed {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def _sv_stats_to_dict(sv: SVStats) -> dict:
    return vars(sv)


def _sv_stats_from_dict(d: dict) -> SVStats:
    return SVStats(**d)


def save_cache(stats: dict[tuple, ConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    for key, cs in stats.items():
        d = {f: getattr(cs, f) for f in ConfigStats.__dataclass_fields__ if f not in ("baseline_svs", "sgd_svs")}
        d["baseline_svs"] = {col: _sv_stats_to_dict(sv) for col, sv in cs.baseline_svs.items()}
        d["sgd_svs"] = {col: _sv_stats_to_dict(sv) for col, sv in cs.sgd_svs.items()}
        cache_data[key] = d
    with open(path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        result = {}
        for key, d in cache_data.items():
            d["baseline_svs"] = {col: _sv_stats_from_dict(sv) for col, sv in d["baseline_svs"].items()}
            d["sgd_svs"] = {col: _sv_stats_from_dict(sv) for col, sv in d["sgd_svs"].items()}
            result[key] = ConfigStats(**d)
        return result
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    exp_config: dict, force_recompute: bool = False,
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


def _build_gamma_max_steps(stats: dict[tuple, ConfigStats]) -> dict[float, int]:
    """Build gamma → max_steps lookup from stats keys (1:1 since they're zipped)."""
    return {key[1]: key[2] for key in stats}


def plot_figure(
    exp_config: dict,
    stats: dict[tuple, ConfigStats],
    gamma_max_steps: dict[float, int],
    k: int,
    hidden_dim: int,
    model_seed: int,
    batch_size: int,
    pp_i: int,
    pp_j: int,
) -> plt.Figure:
    """3 rows (loss, ratio, SVs) × 3 gamma columns."""
    pp_key = f"pp_{pp_i}_{pp_j}_sv"
    product_label = f"W\u2009{pp_j}" if pp_i == pp_j else f"W\u2009{pp_j} \u00b7\u00b7\u00b7 W\u2009{pp_i}"
    baseline_bs = exp_config["baseline_batch_size"]

    bl_label = f"B={baseline_bs}" if baseline_bs is not None else "GD"
    sgd_label = f"B={batch_size}"

    if baseline_bs is not None:
        ratio_ylabel = f"$E(L_{{B={batch_size}}}) \\,/\\, E(L_{{B={baseline_bs}}})$"
    else:
        ratio_ylabel = f"$E(L_{{B={batch_size}}}) \\,/\\, L_{{\\mathrm{{GD}}}}$"

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
    n_runs = 0

    for col, gamma in enumerate(GAMMAS):
        axes[0, col].set_title(f"{GAMMA_NAMES[gamma]} (\u03b3={gamma})", fontsize=11)

        max_steps = gamma_max_steps.get(gamma)
        if max_steps is None:
            continue
        key = (k, gamma, max_steps, hidden_dim, model_seed, batch_size)
        if key not in stats:
            continue

        s = stats[key]
        n_runs = s.n_runs

        # --- Row 0: Loss ---
        ax = axes[0, col]
        ax.plot(s.steps, s.baseline_loss, label=bl_label, color="C0", lw=1.5)
        if s.baseline_spread_lo is not None:
            ax.fill_between(s.steps, s.baseline_spread_lo, s.baseline_spread_hi, alpha=0.15, color="C0")
        ax.plot(s.steps, s.sgd_loss_mean, label=f"E({sgd_label})", color="C1", lw=1.5)
        ax.fill_between(s.steps, s.sgd_loss_spread_lo, s.sgd_loss_spread_hi, alpha=0.2, color="C1")
        ax.fill_between(
            s.steps, 0, 1, where=s.sig_mask, alpha=0.2, color="darkgreen",
            transform=ax.get_xaxis_transform(),
            label=f"{bl_label} < E({sgd_label}) (p<0.05)",
        )
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("Test loss")
        ax.legend(loc="upper right", fontsize=7)

        # --- Row 1: Ratio ---
        ax = axes[1, col]
        ax.axhline(1.0, color="black", ls="--", alpha=0.6, lw=1.2)
        ax.plot(s.steps, s.ratio, color="C1", lw=1.5)
        ax.fill_between(
            s.steps, s.ratio_ci_lo, s.ratio_ci_hi,
            alpha=0.3, color="C1", label="95% CI",
        )
        if col == 0:
            ax.set_ylabel(ratio_ylabel)
        ax.legend(loc="upper right", fontsize=7)

        # --- Row 2: Singular Values ---
        ax = axes[2, col]
        bl_sv = s.baseline_svs.get(pp_key)
        sgd_sv = s.sgd_svs.get(pp_key)
        if bl_sv is not None and sgd_sv is not None:
            n_plot = min(N_PLOT_CAP, bl_sv.mean.shape[1])
            for sv_idx in range(n_plot):
                line, = ax.plot(s.steps, bl_sv.mean[:, sv_idx])
                c = line.get_color()
                if bl_sv.spread_lo is not None:
                    ax.fill_between(
                        s.steps, bl_sv.spread_lo[:, sv_idx], bl_sv.spread_hi[:, sv_idx],
                        alpha=0.08, color=c,
                    )
                ax.plot(s.steps, sgd_sv.mean[:, sv_idx], ls="--", color=c,
                        label=f"\u03c3\u2009{sv_idx}" if sv_idx < 5 else None)
                if sgd_sv.spread_lo is not None:
                    ax.fill_between(
                        s.steps, sgd_sv.spread_lo[:, sv_idx], sgd_sv.spread_hi[:, sv_idx],
                        alpha=0.1, color=c,
                    )
            if col == len(GAMMAS) - 1:
                ax.legend(fontsize=6, ncol=2, loc="center right")

        if col == 0:
            ax.set_ylabel("Singular value")
        ax.set_xlabel("Training step")

    fig.suptitle(
        f"SV dynamics of {product_label}: {bl_label} (solid) vs {sgd_label} (dashed, n={n_runs})\n"
        f"{exp_config['regime_label']} | k={k} | width={hidden_dim} | seed={model_seed}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# Parallel Plot Generation
# =============================================================================

_worker_ctx: dict = {}


def _init_plot_worker(stats: dict, exp_config: dict, gamma_max_steps: dict) -> None:
    _worker_ctx["stats"] = stats
    _worker_ctx["exp"] = exp_config
    _worker_ctx["gms"] = gamma_max_steps


def _run_plot_task(task: tuple) -> None:
    k, hidden_dim, model_seed, batch_size, pp_i, pp_j, out_dir, filename = task
    fig = plot_figure(
        _worker_ctx["exp"], _worker_ctx["stats"], _worker_ctx["gms"],
        k, hidden_dim, model_seed, batch_size, pp_i, pp_j,
    )
    fig.savefig(Path(out_dir) / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(exp_config: dict, stats: dict[tuple, ConfigStats]) -> None:
    figures_path = exp_config["figures_path"]
    figures_path.mkdir(parents=True, exist_ok=True)

    gamma_max_steps = _build_gamma_max_steps(stats)
    all_keys = set(stats.keys())
    k_values = sorted({k[0] for k in all_keys})
    hidden_dims = sorted({k[3] for k in all_keys})
    model_seeds = sorted({k[4] for k in all_keys})
    batch_sizes = sorted({k[5] for k in all_keys})

    tasks = []
    for k in k_values:
        for hidden_dim in hidden_dims:
            for model_seed in model_seeds:
                for batch_size in batch_sizes:
                    for pp_i, pp_j in PARTIAL_PRODUCTS:
                        filename = f"pp_{pp_i}_{pp_j}_k{k}_w{hidden_dim}_mseed{model_seed}_b{batch_size}"
                        tasks.append((
                            k, hidden_dim, model_seed, batch_size,
                            pp_i, pp_j, str(figures_path), filename,
                        ))

    n_workers = min(os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_plot_worker,
        initargs=(stats, exp_config, gamma_max_steps),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_plot_task, tasks), 1):
            print(
                f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="", flush=True,
            )

    print(f"\nAll plots saved to {figures_path}/")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SV dynamics top-k analysis with batch seed averaging",
    )
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
    parser.add_argument(
        "--sort-parquet",
        action="store_true",
        help="Sort parquet files by key columns for efficient reads, then exit",
    )
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    print(f"Experiment: {args.experiment}")
    print(f"Data: {exp['base_path']}")

    if args.sort_parquet:
        sort_parquet(exp, key_cols=BL_KEY_COLS)
        print("Done! Re-run without --sort-parquet to analyze.")
        return

    stats = get_stats(exp, force_recompute=args.recompute)
    generate_all_plots(exp, stats)
    print("\nDone!")


if __name__ == "__main__":
    main()
