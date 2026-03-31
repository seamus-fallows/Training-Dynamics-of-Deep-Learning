"""SV Dynamics Analysis — Top-k Power-Law Teacher

Processes two experiment groups:
  Group A (dim5_topk): Fixed dim=5, varying k (number of active SVs)
  Group B (dim_sweep_k5): Fixed k=5, varying input/output dim

For each group, generates:
  1. SV figures grouped by span: rows = partial products, cols = gamma
     (individual, span-2, span-3, full) — one set per (group_val, width, noise, seed)
  2. Model metrics figures: 2 rows (gram norms, balance diffs) × 3 gamma columns

Usage:
    python analysis/sv_dynamics_topk.py offline
    python analysis/sv_dynamics_topk.py online
    python analysis/sv_dynamics_topk.py offline --recompute
    python analysis/sv_dynamics_topk.py offline --sort-parquet

For best performance on large parquet files, sort them first:
    python analysis/sv_dynamics_topk.py offline --sort-parquet
    python analysis/sv_dynamics_topk.py online --sort-parquet
This enables predicate pushdown so each batch only reads relevant row groups.
"""

import argparse
import gc
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
GRAM_COLS = [f"gram_norm_{i}" for i in range(L)]
BALANCE_COLS = [f"balance_diff_{i}" for i in range(L - 1)]
GAMMAS = sorted(GAMMA_NAMES.keys())

# Partial product groupings (matching sv_dynamics.py)
INDIVIDUAL = [(i, i) for i in range(L)]
COMPOSITE_FIGURES = [
    ("span2", [(0, 1), (1, 2), (2, 3)]),
    ("span3", [(0, 2), (1, 3)]),
    ("full", [(0, 3)]),
]

SV_COLORS = [f"C{i}" for i in range(N_PLOT_CAP)]

BASE_PATH = Path("outputs/sv_dynamics_topk")

GROUPS = {
    "dim5_topk": {
        "subdir": "dim5_topk",
        "group_col": "data.params.k",
        "group_tag": "k",
        "bl_key_cols": [
            "data.params.k", "model.gamma", "max_steps",
            "model.hidden_dim", "data.noise_std", "model.model_seed",
        ],
    },
    "dim_sweep_k5": {
        "subdir": "dim_sweep_k5",
        "group_col": "model.in_dim",
        "group_tag": "dim",
        "bl_key_cols": [
            "model.in_dim", "model.gamma", "max_steps",
            "model.hidden_dim", "data.noise_std", "model.model_seed",
        ],
    },
}

REGIMES = {
    "offline": {
        "baseline_subdir": "offline/full_batch",
        "sgd_subdir": "offline/mini_batch",
        "baseline_label": "GD",
        "baseline_batch_size": None,
        "sgd_batch_sizes": [1],
        "regime_label": "Offline (N=500 samples)",
    },
    "online": {
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
class LayerStats:
    """Per-layer scalar metric statistics: arrays of shape (n_steps,)."""
    mean: np.ndarray
    n: int
    ci_lo: np.ndarray | None  # None for n=1
    ci_hi: np.ndarray | None


@dataclass
class ConfigStats:
    """All plotting data for one configuration."""
    steps: np.ndarray
    n_runs: int
    # SVs per partial product
    baseline_svs: dict[str, SVStats] = field(default_factory=dict)
    sgd_svs: dict[str, SVStats] = field(default_factory=dict)
    # Model metrics per layer
    baseline_gram_norms: dict[int, LayerStats] = field(default_factory=dict)
    sgd_gram_norms: dict[int, LayerStats] = field(default_factory=dict)
    baseline_balance_diffs: dict[int, LayerStats] = field(default_factory=dict)
    sgd_balance_diffs: dict[int, LayerStats] = field(default_factory=dict)


# =============================================================================
# Data Loading & Statistics
# =============================================================================


def _extract_sv_3d(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract SV data as (n_runs, n_steps, n_svs) array, capped to N_PLOT_CAP SVs."""
    return np.stack([np.array(row)[:, :N_PLOT_CAP] for row in df[col].to_list()])


def _make_sv_stats(curves_3d: np.ndarray) -> SVStats:
    """Compute mean and spread for a (n_runs, n_steps, n_svs) array."""
    n_runs, n_steps, n_svs = curves_3d.shape
    mean = curves_3d.mean(axis=0)

    if n_runs == 1:
        return SVStats(mean=mean, n=1, spread_lo=None, spread_hi=None)

    flat = curves_3d.reshape(n_runs, -1)
    flat_mean = mean.reshape(-1)
    lo, hi = mean_centered_spread(flat, flat_mean)
    return SVStats(
        mean=mean, n=n_runs,
        spread_lo=lo.reshape(n_steps, n_svs),
        spread_hi=hi.reshape(n_steps, n_svs),
    )


def _make_layer_stats(df: pl.DataFrame, col: str) -> LayerStats:
    """Compute mean and 95% CI for a layer metric column."""
    curves = np.vstack(df[col].to_list())
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return LayerStats(mean=mean, n=1, ci_lo=None, ci_hi=None)

    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(curves.var(axis=0, ddof=1) / n)
    return LayerStats(mean=mean, n=n, ci_lo=mean - t_val * sem, ci_hi=mean + t_val * sem)


@dataclass
class _BaselineStats:
    """Intermediate baseline data for computing ConfigStats."""
    steps: np.ndarray
    svs: dict[str, SVStats]
    gram_norms: dict[int, LayerStats]
    balance_diffs: dict[int, LayerStats]


def _compute_baseline(
    group_df: pl.DataFrame,
    available_gram: list[str],
    available_balance: list[str],
) -> _BaselineStats:
    steps = np.array(group_df["step"][0])

    svs = {}
    for col in PP_SV_COLS:
        if col in group_df.columns:
            svs[col] = _make_sv_stats(_extract_sv_3d(group_df, col))

    gram_norms = {}
    for col in available_gram:
        i = int(col.rsplit("_", 1)[1])
        gram_norms[i] = _make_layer_stats(group_df, col)

    balance_diffs = {}
    for col in available_balance:
        i = int(col.rsplit("_", 1)[1])
        balance_diffs[i] = _make_layer_stats(group_df, col)

    return _BaselineStats(
        steps=steps, svs=svs,
        gram_norms=gram_norms, balance_diffs=balance_diffs,
    )


def _compute_config_stats(
    sgd_df: pl.DataFrame,
    bl: _BaselineStats,
    available_gram: list[str],
    available_balance: list[str],
) -> ConfigStats:
    n = len(sgd_df)

    sgd_svs = {}
    for col in PP_SV_COLS:
        if col in sgd_df.columns:
            sgd_svs[col] = _make_sv_stats(_extract_sv_3d(sgd_df, col))

    sgd_gram_norms = {}
    for col in available_gram:
        i = int(col.rsplit("_", 1)[1])
        sgd_gram_norms[i] = _make_layer_stats(sgd_df, col)

    sgd_balance_diffs = {}
    for col in available_balance:
        i = int(col.rsplit("_", 1)[1])
        sgd_balance_diffs[i] = _make_layer_stats(sgd_df, col)

    return ConfigStats(
        steps=bl.steps, n_runs=n,
        baseline_svs=bl.svs, sgd_svs=sgd_svs,
        baseline_gram_norms=bl.gram_norms, sgd_gram_norms=sgd_gram_norms,
        baseline_balance_diffs=bl.balance_diffs, sgd_balance_diffs=sgd_balance_diffs,
    )


# =============================================================================
# Statistics Computation
# =============================================================================


def compute_group_stats(
    group_cfg: dict, regime_cfg: dict,
) -> dict[tuple, ConfigStats]:
    bl_key_cols = group_cfg["bl_key_cols"]
    batch_key_cols = bl_key_cols[:-1]  # everything except model_seed

    base_dir = BASE_PATH / group_cfg["subdir"]
    baseline_dir = base_dir / regime_cfg["baseline_subdir"]
    sgd_dir = base_dir / regime_cfg["sgd_subdir"]

    # Discover available metric columns from parquet schema
    bl_schema = pl.scan_parquet(baseline_dir / "results.parquet").collect_schema().names()
    available_gram = [c for c in GRAM_COLS if c in bl_schema]
    available_balance = [c for c in BALANCE_COLS if c in bl_schema]
    metric_cols = available_gram + available_balance

    available_pp = [c for c in PP_SV_COLS if c in bl_schema]
    bl_select = bl_key_cols + ["step"] + available_pp + metric_cols

    sgd_schema = pl.scan_parquet(sgd_dir / "results.parquet").collect_schema()
    has_bs_col = "training.batch_size" in sgd_schema.names()
    sgd_available_pp = [c for c in PP_SV_COLS if c in sgd_schema.names()]
    sgd_metric_cols = [c for c in metric_cols if c in sgd_schema.names()]
    sgd_select = (
        bl_key_cols
        + (["training.batch_size"] if has_bs_col else [])
        + sgd_available_pp + sgd_metric_cols
    )

    # Phase 1: Baselines — load, compute stats, free DF
    print(f"  Loading baselines from {baseline_dir}...")
    bl_df = (
        pl.scan_parquet(baseline_dir / "results.parquet")
        .select(bl_select)
        .collect(engine="streaming")
    )
    bl_groups = bl_df.partition_by(bl_key_cols, as_dict=True)
    del bl_df
    gc.collect()

    print(f"  Computing baselines ({len(bl_groups)} groups)...")
    baselines: dict[tuple, _BaselineStats] = {}
    for key, group_df in bl_groups.items():
        if len(group_df) > 0:
            baselines[key] = _compute_baseline(group_df, available_gram, available_balance)
    del bl_groups
    gc.collect()
    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — batch by everything except model_seed.
    # Each batch filter+collects one (group_val, gamma, max_steps, hidden_dim, noise)
    # combination from the parquet, keeping memory bounded.
    # For best performance, sort the parquet first with --sort-parquet.
    sgd_lf = pl.scan_parquet(sgd_dir / "results.parquet")
    sgd_batch_sizes = regime_cfg["sgd_batch_sizes"]

    key_batches: dict[tuple, list[tuple]] = defaultdict(list)
    for bl_key in sorted(baselines):
        key_batches[bl_key[:-1]].append(bl_key)

    n_batches = len(key_batches)
    total_configs = len(baselines)
    print(f"  Computing SGD statistics ({n_batches} batches, {total_configs} baseline groups)...")

    stats: dict[tuple, ConfigStats] = {}
    completed = 0
    for batch_key, batch_bl_keys in key_batches.items():
        chunk = (
            sgd_lf.filter(build_filter(batch_key_cols, batch_key))
            .select(sgd_select)
            .collect(engine="streaming")
        )
        if has_bs_col:
            sub_groups = chunk.partition_by(
                ["model.model_seed", "training.batch_size"], as_dict=True,
            )
        else:
            sub_groups = {
                (ms, sgd_batch_sizes[0]): g
                for (ms,), g in chunk.partition_by(
                    ["model.model_seed"], as_dict=True,
                ).items()
            }
        del chunk

        for (model_seed, batch_size), group_df in sub_groups.items():
            bl_key = batch_key + (model_seed,)
            bl = baselines.get(bl_key)
            if bl is None or len(group_df) == 0:
                continue
            key = bl_key + (batch_size,)
            stats[key] = _compute_config_stats(
                group_df, bl, available_gram, available_balance,
            )

        completed += len(batch_bl_keys)
        print(
            f"\r  SGD stats: {completed}/{total_configs}"
            f" ({100 * completed / total_configs:.0f}%)",
            end="", flush=True,
        )
        del sub_groups

    del baselines
    gc.collect()
    print(f"\n  Computed {len(stats)} configurations")
    return stats


# =============================================================================
# Caching
# =============================================================================


def _to_dict(obj) -> dict:
    return vars(obj)


def _sv_from_dict(d: dict) -> SVStats:
    return SVStats(**d)


def _layer_from_dict(d: dict) -> LayerStats:
    return LayerStats(**d)


def save_cache(stats: dict[tuple, ConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    for key, cs in stats.items():
        d = {
            f: getattr(cs, f)
            for f in ConfigStats.__dataclass_fields__
            if f not in (
                "baseline_svs", "sgd_svs",
                "baseline_gram_norms", "sgd_gram_norms",
                "baseline_balance_diffs", "sgd_balance_diffs",
            )
        }
        d["baseline_svs"] = {col: _to_dict(sv) for col, sv in cs.baseline_svs.items()}
        d["sgd_svs"] = {col: _to_dict(sv) for col, sv in cs.sgd_svs.items()}
        d["baseline_gram_norms"] = {i: _to_dict(ls) for i, ls in cs.baseline_gram_norms.items()}
        d["sgd_gram_norms"] = {i: _to_dict(ls) for i, ls in cs.sgd_gram_norms.items()}
        d["baseline_balance_diffs"] = {i: _to_dict(ls) for i, ls in cs.baseline_balance_diffs.items()}
        d["sgd_balance_diffs"] = {i: _to_dict(ls) for i, ls in cs.sgd_balance_diffs.items()}
        cache_data[key] = d
    with open(path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        result = {}
        for key, d in cache_data.items():
            d["baseline_svs"] = {col: _sv_from_dict(sv) for col, sv in d["baseline_svs"].items()}
            d["sgd_svs"] = {col: _sv_from_dict(sv) for col, sv in d["sgd_svs"].items()}
            d["baseline_gram_norms"] = {i: _layer_from_dict(ls) for i, ls in d["baseline_gram_norms"].items()}
            d["sgd_gram_norms"] = {i: _layer_from_dict(ls) for i, ls in d["sgd_gram_norms"].items()}
            d["baseline_balance_diffs"] = {i: _layer_from_dict(ls) for i, ls in d["baseline_balance_diffs"].items()}
            d["sgd_balance_diffs"] = {i: _layer_from_dict(ls) for i, ls in d["sgd_balance_diffs"].items()}
            result[key] = ConfigStats(**d)
        return result
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"  Warning: Failed to load cache ({e}), will recompute")
        return None


def get_group_stats(
    group_name: str,
    group_cfg: dict,
    regime_name: str,
    regime_cfg: dict,
    force_recompute: bool = False,
) -> dict[tuple, ConfigStats]:
    cache_path = CACHE_DIR / f"sv_dynamics_topk_{group_name}_{regime_name}_v2.pkl"
    if not force_recompute:
        stats = load_cache(cache_path)
        if stats is not None:
            print(f"  Loaded {len(stats)} configurations from cache")
            return stats
    stats = compute_group_stats(group_cfg, regime_cfg)
    save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting Helpers
# =============================================================================


def _pp_label(i, j):
    if i == j:
        return f"$W_{{{j}}}$"
    layers = " ".join(f"W_{{{k}}}" for k in range(j, i - 1, -1))
    return f"${layers}$"


def _pp_ylabel(i, j, n_total_svs=None):
    label = _pp_label(i, j)
    if n_total_svs is not None and n_total_svs >= N_PLOT_CAP:
        return f"Top {N_PLOT_CAP} SVs of {label}"
    return f"SVs of {label}"


def _plot_sv_panel(ax, steps, bl_sv, sgd_sv):
    """Plot baseline (solid) and SGD (dashed) SVs on a single axis."""
    n_plot = min(N_PLOT_CAP, bl_sv.mean.shape[1])
    for k in range(n_plot):
        ax.plot(steps, bl_sv.mean[:, k], color=SV_COLORS[k], linewidth=1.2)
        ax.plot(
            steps, sgd_sv.mean[:, k], color=SV_COLORS[k],
            linewidth=1.2, linestyle="--",
        )


def _build_gamma_max_steps(stats: dict[tuple, ConfigStats]) -> dict[float, int]:
    """Build gamma -> max_steps lookup from stats keys (1:1 since they're zipped)."""
    return {key[1]: key[2] for key in stats}


# =============================================================================
# Plotting — SV Figures
# =============================================================================


def make_sv_figure(
    stats: dict[tuple, ConfigStats],
    gamma_max_steps: dict[float, int],
    products: list[tuple[int, int]],
    group_val: int,
    hidden_dim: int,
    noise: float,
    model_seed: int,
    batch_size: int,
    ref_label: str,
    sgd_label: str,
) -> plt.Figure:
    """Rows = partial products, columns = gamma. Baseline solid, SGD dashed."""
    n_rows = len(products)
    n_cols = len(GAMMAS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), squeeze=False,
    )

    for col, gamma in enumerate(GAMMAS):
        max_steps = gamma_max_steps.get(gamma)
        if max_steps is None:
            continue
        key = (group_val, gamma, max_steps, hidden_dim, noise, model_seed, batch_size)
        if key not in stats:
            continue

        s = stats[key]

        for row, (i, j) in enumerate(products):
            ax = axes[row, col]
            pp_col = f"pp_{i}_{j}_sv"
            bl_sv = s.baseline_svs.get(pp_col)
            sgd_sv = s.sgd_svs.get(pp_col)
            if bl_sv is not None and sgd_sv is not None:
                _plot_sv_panel(ax, s.steps, bl_sv, sgd_sv)
                if col == 0:
                    ax.set_ylabel(
                        _pp_ylabel(i, j, bl_sv.mean.shape[1]), fontsize=10,
                    )
            if row == 0:
                ax.set_title(GAMMA_NAMES[gamma], fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color="gray", linewidth=1.2, label=ref_label),
        Line2D([], [], color="gray", linewidth=1.2, linestyle="--", label=sgd_label),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    return fig


# =============================================================================
# Plotting — Model Metrics Figures
# =============================================================================


def _plot_ref_vs_sgd_layers(
    ax: plt.Axes,
    steps: np.ndarray,
    ref_dict: dict[int, LayerStats],
    sgd_dict: dict[int, LayerStats],
    ylabel: str,
    ref_label: str,
    sgd_label: str,
    show_legend: bool = False,
) -> None:
    """Plot per-layer ref (solid) vs SGD (dashed + CI) comparison."""
    for i in sorted(ref_dict.keys()):
        color = f"C{i}"
        ref = ref_dict[i]
        ax.plot(steps, ref.mean, color=color, linewidth=1.5, label=f"L{i} {ref_label}")
        if ref.ci_lo is not None:
            ax.fill_between(steps, ref.ci_lo, ref.ci_hi, alpha=0.1, color=color)
        if i in sgd_dict:
            sgd = sgd_dict[i]
            ax.plot(
                steps, sgd.mean, color=color, linewidth=1.5,
                linestyle="--", label=f"L{i} {sgd_label}",
            )
            if sgd.ci_lo is not None:
                ax.fill_between(steps, sgd.ci_lo, sgd.ci_hi, alpha=0.15, color=color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if show_legend:
        ax.legend(fontsize=7, loc="best", ncols=2)


def make_model_metrics_figure(
    stats: dict[tuple, ConfigStats],
    gamma_max_steps: dict[float, int],
    group_val: int,
    hidden_dim: int,
    noise: float,
    model_seed: int,
    batch_size: int,
    ref_label: str,
    sgd_label: str,
    group_tag: str,
    regime_label: str,
) -> plt.Figure:
    """2 rows (gram norms, balance diffs) x 3 gamma columns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), squeeze=False)
    n_runs = 0

    for col, gamma in enumerate(GAMMAS):
        axes[0, col].set_title(f"{GAMMA_NAMES[gamma]} (\u03b3={gamma})", fontsize=10)

        max_steps = gamma_max_steps.get(gamma)
        if max_steps is None:
            continue
        key = (group_val, gamma, max_steps, hidden_dim, noise, model_seed, batch_size)
        if key not in stats:
            continue

        s = stats[key]
        n_runs = s.n_runs
        last_col = col == len(GAMMAS) - 1

        # Row 0: Gram norms
        _plot_ref_vs_sgd_layers(
            axes[0, col], s.steps,
            s.baseline_gram_norms, s.sgd_gram_norms,
            ylabel=r"$\|W_i W_i^T\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_label, show_legend=last_col,
        )

        # Row 1: Balance diffs
        _plot_ref_vs_sgd_layers(
            axes[1, col], s.steps,
            s.baseline_balance_diffs, s.sgd_balance_diffs,
            ylabel=r"$\|G_l\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_label, show_legend=last_col,
        )
        axes[1, col].set_xlabel("Training step")

    fig.suptitle(
        f"Model metrics: {ref_label} (solid) vs {sgd_label} (dashed, n={n_runs})\n"
        f"{regime_label} | {group_tag}={group_val} | noise={noise} "
        f"| width={hidden_dim} | seed={model_seed}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# Plot Generation
# =============================================================================


def generate_group_plots(
    group_name: str,
    group_cfg: dict,
    regime_name: str,
    regime_cfg: dict,
    stats: dict[tuple, ConfigStats],
) -> None:
    figures_base = Path("figures/sv_dynamics_topk") / regime_name / group_name
    gamma_max_steps = _build_gamma_max_steps(stats)
    tag = group_cfg["group_tag"]

    baseline_bs = regime_cfg["baseline_batch_size"]
    ref_label = f"B={baseline_bs}" if baseline_bs is not None else "GD"

    # Extract unique parameter values from stats keys
    # Key: (group_val, gamma, max_steps, hidden_dim, noise, model_seed, batch_size)
    all_keys = set(stats.keys())
    group_vals = sorted({k[0] for k in all_keys})
    hidden_dims = sorted({k[3] for k in all_keys})
    noise_levels = sorted({k[4] for k in all_keys})
    model_seeds = sorted({k[5] for k in all_keys})
    batch_sizes = sorted({k[6] for k in all_keys})

    # Count total figures for progress reporting
    n_configs = (len(group_vals) * len(noise_levels) * len(hidden_dims)
                 * len(model_seeds) * len(batch_sizes))
    figs_per_config = 1 + 1 + len(COMPOSITE_FIGURES)  # metrics + individual + composites
    total = n_configs * figs_per_config
    done = 0

    print(f"  Generating {total} figures...")

    for group_val in group_vals:
        for noise in noise_levels:
            noise_dir = figures_base / f"noise_{noise}"
            noise_dir.mkdir(parents=True, exist_ok=True)
            for hidden_dim in hidden_dims:
                for model_seed in model_seeds:
                    for batch_size in batch_sizes:
                        base_name = (
                            f"{tag}{group_val}_w{hidden_dim}"
                            f"_mseed{model_seed}_b{batch_size}"
                        )
                        sgd_label = f"B={batch_size}"

                        # Model metrics figure
                        fig = make_model_metrics_figure(
                            stats, gamma_max_steps,
                            group_val, hidden_dim, noise, model_seed, batch_size,
                            ref_label, sgd_label,
                            group_tag=tag,
                            regime_label=regime_cfg["regime_label"],
                        )
                        fig.savefig(
                            noise_dir / f"model_metrics_{base_name}.png",
                            dpi=150, bbox_inches="tight",
                        )
                        plt.close(fig)
                        done += 1

                        # SV figures grouped by span
                        for slug, products in [("individual", INDIVIDUAL)] + COMPOSITE_FIGURES:
                            fig = make_sv_figure(
                                stats, gamma_max_steps, products,
                                group_val, hidden_dim, noise,
                                model_seed, batch_size,
                                ref_label, sgd_label,
                            )
                            fig.savefig(
                                noise_dir / f"{slug}_{base_name}.png",
                                dpi=150, bbox_inches="tight",
                            )
                            plt.close(fig)
                            done += 1

                        if done % 20 == 0 or done == total:
                            print(
                                f"\r  Progress: {done}/{total}"
                                f" ({100 * done / total:.0f}%)",
                                end="", flush=True,
                            )

    print(f"\n  Plots saved to {figures_base}/")


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
        help="Which experiment regime to analyze",
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
    parser.add_argument(
        "--group",
        choices=list(GROUPS.keys()),
        default=None,
        help="Process only this group (default: both)",
    )
    args = parser.parse_args()

    regime_name = args.experiment
    regime_cfg = REGIMES[regime_name]
    groups_to_run = (
        {args.group: GROUPS[args.group]} if args.group else GROUPS
    )

    print(f"Experiment: {regime_name}")

    if args.sort_parquet:
        for group_name, group_cfg in groups_to_run.items():
            print(f"\n=== Sorting {group_name} ===")
            sort_config = {
                "base_path": BASE_PATH / group_cfg["subdir"],
                "baseline_subdir": regime_cfg["baseline_subdir"],
                "sgd_subdir": regime_cfg["sgd_subdir"],
            }
            sort_parquet(sort_config, key_cols=group_cfg["bl_key_cols"])
        print("\nDone! Re-run without --sort-parquet to analyze.")
        return

    for group_name, group_cfg in groups_to_run.items():
        print(f"\n=== {group_name} ({regime_name}) ===")
        stats = get_group_stats(
            group_name, group_cfg, regime_name, regime_cfg,
            force_recompute=args.recompute,
        )
        generate_group_plots(group_name, group_cfg, regime_name, regime_cfg, stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
