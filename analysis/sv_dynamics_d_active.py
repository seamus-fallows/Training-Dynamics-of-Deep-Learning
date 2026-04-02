"""SV Dynamics Analysis — d_active Experiment

Processes outputs from the sv_dynamics_d_active sweep, which varies
input/output dimensions (in_dim = out_dim ∈ {6, 8, 10}) with a fixed
d_active=5 power-law teacher (5 active singular values).

Generates per-(in_dim, width, seed, batch_size) configuration:
  1. SV figures grouped by span: rows = partial products, cols = gamma
     (individual, span-2, span-3, full)
  2. Model metrics figures: 2 rows (gram norms, balance diffs) × 3 gamma columns

Usage:
    python analysis/sv_dynamics_d_active.py offline
    python analysis/sv_dynamics_d_active.py online
    python analysis/sv_dynamics_d_active.py offline --recompute
    python analysis/sv_dynamics_d_active.py offline --sort-parquet
"""

import argparse
import gc
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

import pyarrow.parquet as pq

from _cache import save_cache as _save_raw, load_cache as _load_raw
from _common import (
    CACHE_DIR, GAMMA_NAMES,
    extract_curves, mean_centered_spread, pp_label, sort_parquet,
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

INDIVIDUAL = [(i, i) for i in range(L)]
COMPOSITE_FIGURES = [
    ("span2", [(0, 1), (1, 2), (2, 3)]),
    ("span3", [(0, 2), (1, 3)]),
    ("full", [(0, 3)]),
]

SV_COLORS = [f"C{i}" for i in range(N_PLOT_CAP)]

BASE_PATH = Path("outputs/sv_dynamics_d_active")

# Key columns: in_dim identifies the configuration (out_dim is zipped with it)
BL_KEY_COLS = [
    "model.in_dim", "model.gamma", "max_steps",
    "model.hidden_dim", "model.model_seed",
]

GROUP_TAG = "dim"

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
    spread_lo: np.ndarray | None = None
    spread_hi: np.ndarray | None = None


@dataclass
class LayerStats:
    """Per-layer scalar metric statistics: arrays of shape (n_steps,)."""
    mean: np.ndarray
    n: int
    ci_lo: np.ndarray | None = None
    ci_hi: np.ndarray | None = None


@dataclass
class ConfigStats:
    """All plotting data for one configuration."""
    steps: np.ndarray
    n_runs: int
    baseline_svs: dict[str, SVStats] = field(default_factory=dict)
    sgd_svs: dict[str, SVStats] = field(default_factory=dict)
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
    n_runs, n_steps, n_svs = curves_3d.shape
    mean = curves_3d.mean(axis=0)

    if n_runs == 1:
        return SVStats(mean=mean, n=1)

    flat = curves_3d.reshape(n_runs, -1)
    flat_mean = mean.reshape(-1)
    lo, hi = mean_centered_spread(flat, flat_mean)
    return SVStats(
        mean=mean, n=n_runs,
        spread_lo=lo.reshape(n_steps, n_svs),
        spread_hi=hi.reshape(n_steps, n_svs),
    )


def _make_layer_stats(df: pl.DataFrame, col: str) -> LayerStats:
    curves = extract_curves(df, col)
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return LayerStats(mean=mean, n=1)

    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(curves.var(axis=0, ddof=1) / n)
    return LayerStats(mean=mean, n=n, ci_lo=mean - t_val * sem, ci_hi=mean + t_val * sem)


@dataclass
class _BaselineStats:
    steps: np.ndarray
    svs: dict[str, SVStats]
    gram_norms: dict[int, LayerStats]
    balance_diffs: dict[int, LayerStats]


# =============================================================================
# Statistics Computation
# =============================================================================


def _stream_partition(
    path: Path,
    columns: list[str],
    partition_cols: list[str],
    batch_size: int = 500,
) -> dict[tuple, pl.DataFrame]:
    """Stream a parquet file and partition into groups.

    Reads in small batches via pyarrow to bound peak memory, then
    partitions and accumulates rows per group key. Returns concatenated
    DataFrames per group.
    """
    pf = pq.ParquetFile(path)
    groups: dict[tuple, list[pl.DataFrame]] = defaultdict(list)
    for record_batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        chunk = pl.from_arrow(record_batch)
        for key, sub_df in chunk.partition_by(partition_cols, as_dict=True).items():
            groups[key].append(sub_df)
    return {key: pl.concat(dfs) for key, dfs in groups.items()}


def compute_stats(regime_cfg: dict) -> dict[tuple, ConfigStats]:
    baseline_dir = BASE_PATH / regime_cfg["baseline_subdir"]
    sgd_dir = BASE_PATH / regime_cfg["sgd_subdir"]

    # Discover available metric columns
    bl_schema = pl.scan_parquet(baseline_dir / "results.parquet").collect_schema().names()
    available_gram = [c for c in GRAM_COLS if c in bl_schema]
    available_balance = [c for c in BALANCE_COLS if c in bl_schema]
    metric_cols = available_gram + available_balance

    available_pp = [c for c in PP_SV_COLS if c in bl_schema]

    sgd_schema = pl.scan_parquet(sgd_dir / "results.parquet").collect_schema()
    has_bs_col = "training.batch_size" in sgd_schema.names()
    sgd_available_pp = [c for c in PP_SV_COLS if c in sgd_schema.names()]
    sgd_metric_cols = [c for c in metric_cols if c in sgd_schema.names()]

    # Phase 1: Baselines — column-split streaming to bound peak memory.
    bl_path = baseline_dir / "results.parquet"
    print(f"  Loading baselines from {baseline_dir}...")

    # Pass 1a: steps + metric columns (lightweight)
    bl_metric_parts = _stream_partition(
        bl_path, BL_KEY_COLS + ["step"] + metric_cols, BL_KEY_COLS,
    )

    baselines: dict[tuple, _BaselineStats] = {}
    for key, group_df in bl_metric_parts.items():
        if len(group_df) == 0:
            continue
        steps = np.array(group_df["step"][0])
        gram_norms = {}
        for col in available_gram:
            i = int(col.rsplit("_", 1)[1])
            gram_norms[i] = _make_layer_stats(group_df, col)
        balance_diffs = {}
        for col in available_balance:
            i = int(col.rsplit("_", 1)[1])
            balance_diffs[i] = _make_layer_stats(group_df, col)
        baselines[key] = _BaselineStats(
            steps=steps, svs={}, gram_norms=gram_norms, balance_diffs=balance_diffs,
        )
    del bl_metric_parts
    gc.collect()

    # Pass 1b+: SV columns one at a time
    for pp_col in available_pp:
        bl_sv_parts = _stream_partition(
            bl_path, BL_KEY_COLS + [pp_col], BL_KEY_COLS,
        )
        for key, group_df in bl_sv_parts.items():
            if key in baselines:
                baselines[key].svs[pp_col] = _make_sv_stats(
                    _extract_sv_3d(group_df, pp_col)
                )
        del bl_sv_parts
        gc.collect()

    print(f"  {len(baselines)} baselines computed")

    # Phase 2: SGD — column-split streaming to bound memory.
    sgd_path = sgd_dir / "results.parquet"
    sgd_batch_sizes = regime_cfg["sgd_batch_sizes"]
    partition_cols = BL_KEY_COLS + (["training.batch_size"] if has_bs_col else [])

    def _resolve_key(key):
        if has_bs_col:
            return key[:-1], key[-1]
        return key, sgd_batch_sizes[0]

    # Pass 2a: Metric stats (lightweight — no nested SV arrays)
    print(f"  Computing metric stats from {sgd_dir}...")
    metric_parts = _stream_partition(
        sgd_path, partition_cols + sgd_metric_cols, partition_cols,
    )

    stats: dict[tuple, ConfigStats] = {}
    for key, group_df in metric_parts.items():
        bl_key, batch_size = _resolve_key(key)
        bl = baselines.get(bl_key)
        if bl is None or len(group_df) == 0:
            continue

        sgd_gram = {}
        for col in available_gram:
            if col in sgd_metric_cols:
                i = int(col.rsplit("_", 1)[1])
                sgd_gram[i] = _make_layer_stats(group_df, col)

        sgd_balance = {}
        for col in available_balance:
            if col in sgd_metric_cols:
                i = int(col.rsplit("_", 1)[1])
                sgd_balance[i] = _make_layer_stats(group_df, col)

        stats_key = bl_key + (batch_size,)
        stats[stats_key] = ConfigStats(
            steps=bl.steps, n_runs=len(group_df),
            baseline_svs=bl.svs, sgd_svs={},
            baseline_gram_norms=bl.gram_norms, sgd_gram_norms=sgd_gram,
            baseline_balance_diffs=bl.balance_diffs, sgd_balance_diffs=sgd_balance,
        )
    del metric_parts
    gc.collect()
    print(f"  {len(stats)} configs with metric stats")

    # Pass 2b+: SV stats (one PP column per pass to bound memory)
    n_pp = len(sgd_available_pp)
    for pp_idx, pp_col in enumerate(sgd_available_pp):
        print(
            f"\r  SV column {pp_idx + 1}/{n_pp}: {pp_col}",
            end="", flush=True,
        )
        sv_parts = _stream_partition(
            sgd_path, partition_cols + [pp_col], partition_cols,
        )
        for key, group_df in sv_parts.items():
            bl_key, batch_size = _resolve_key(key)
            stats_key = bl_key + (batch_size,)
            if stats_key in stats:
                stats[stats_key].sgd_svs[pp_col] = _make_sv_stats(
                    _extract_sv_3d(group_df, pp_col)
                )
        del sv_parts
        gc.collect()

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
    _save_raw(cache_data, path)
    print(f"  Cache saved to {path}")


def load_cache(path: Path) -> dict[tuple, ConfigStats] | None:
    cache_data = _load_raw(path)
    if cache_data is None:
        return None
    try:
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
    except (KeyError, TypeError) as e:
        print(f"  Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    regime_name: str,
    regime_cfg: dict,
    force_recompute: bool = False,
) -> dict[tuple, ConfigStats]:
    cache_path = CACHE_DIR / f"sv_dynamics_d_active_{regime_name}.pkl"
    if not force_recompute:
        stats = load_cache(cache_path)
        if stats is not None:
            print(f"  Loaded {len(stats)} configurations from cache")
            return stats
    stats = compute_stats(regime_cfg)
    save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting Helpers
# =============================================================================


def _pp_ylabel(i, j, n_total_svs=None):
    label = pp_label(i, j)
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
    """Build gamma -> max_steps lookup from stats keys."""
    return {key[1]: key[2] for key in stats}


# =============================================================================
# Plotting — SV Figures
# =============================================================================


def make_sv_figure(
    stats: dict[tuple, ConfigStats],
    gamma_max_steps: dict[float, int],
    products: list[tuple[int, int]],
    in_dim: int,
    hidden_dim: int,
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
        key = (in_dim, gamma, max_steps, hidden_dim, model_seed, batch_size)
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
    in_dim: int,
    hidden_dim: int,
    model_seed: int,
    batch_size: int,
    ref_label: str,
    sgd_label: str,
    regime_label: str,
) -> plt.Figure:
    """2 rows (gram norms, balance diffs) × 3 gamma columns."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), squeeze=False)
    n_runs = 0

    for col, gamma in enumerate(GAMMAS):
        axes[0, col].set_title(f"{GAMMA_NAMES[gamma]} (γ={gamma})", fontsize=10)

        max_steps = gamma_max_steps.get(gamma)
        if max_steps is None:
            continue
        key = (in_dim, gamma, max_steps, hidden_dim, model_seed, batch_size)
        if key not in stats:
            continue

        s = stats[key]
        n_runs = s.n_runs
        last_col = col == len(GAMMAS) - 1

        _plot_ref_vs_sgd_layers(
            axes[0, col], s.steps,
            s.baseline_gram_norms, s.sgd_gram_norms,
            ylabel=r"$\|W_i W_i^T\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_label, show_legend=last_col,
        )

        _plot_ref_vs_sgd_layers(
            axes[1, col], s.steps,
            s.baseline_balance_diffs, s.sgd_balance_diffs,
            ylabel=r"$\|G_l\|_F$" if col == 0 else "",
            ref_label=ref_label, sgd_label=sgd_label, show_legend=last_col,
        )
        axes[1, col].set_xlabel("Training step")

    fig.suptitle(
        f"Model metrics: {ref_label} (solid) vs {sgd_label} (dashed, n={n_runs})\n"
        f"{regime_label} | in_dim=out_dim={in_dim} | d_active=5 "
        f"| width={hidden_dim} | seed={model_seed}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
# Plot Generation
# =============================================================================


def generate_plots(
    regime_name: str,
    regime_cfg: dict,
    stats: dict[tuple, ConfigStats],
) -> None:
    figures_base = Path("figures/sv_dynamics_d_active") / regime_name
    gamma_max_steps = _build_gamma_max_steps(stats)

    baseline_bs = regime_cfg["baseline_batch_size"]
    ref_label = f"B={baseline_bs}" if baseline_bs is not None else "GD"

    # Key: (in_dim, gamma, max_steps, hidden_dim, model_seed, batch_size)
    all_keys = set(stats.keys())
    in_dims = sorted({k[0] for k in all_keys})
    hidden_dims = sorted({k[3] for k in all_keys})
    model_seeds = sorted({k[4] for k in all_keys})
    batch_sizes = sorted({k[5] for k in all_keys})

    n_configs = len(in_dims) * len(hidden_dims) * len(model_seeds) * len(batch_sizes)
    figs_per_config = 1 + 1 + len(COMPOSITE_FIGURES)  # metrics + individual + composites
    total = n_configs * figs_per_config
    done = 0

    print(f"  Generating {total} figures...")

    for in_dim in in_dims:
        for hidden_dim in hidden_dims:
            fig_dir = figures_base / f"dim{in_dim}"
            fig_dir.mkdir(parents=True, exist_ok=True)

            for model_seed in model_seeds:
                for batch_size in batch_sizes:
                    base_name = f"dim{in_dim}_w{hidden_dim}_mseed{model_seed}_b{batch_size}"
                    sgd_label = f"B={batch_size}"

                    # Model metrics figure
                    fig = make_model_metrics_figure(
                        stats, gamma_max_steps,
                        in_dim, hidden_dim, model_seed, batch_size,
                        ref_label, sgd_label,
                        regime_label=regime_cfg["regime_label"],
                    )
                    fig.savefig(
                        fig_dir / f"model_metrics_{base_name}.png",
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)
                    done += 1

                    # SV figures grouped by span
                    for slug, products in [("individual", INDIVIDUAL)] + COMPOSITE_FIGURES:
                        fig = make_sv_figure(
                            stats, gamma_max_steps, products,
                            in_dim, hidden_dim, model_seed, batch_size,
                            ref_label, sgd_label,
                        )
                        fig.savefig(
                            fig_dir / f"{slug}_{base_name}.png",
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
        description="SV dynamics analysis for d_active experiment",
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
    args = parser.parse_args()

    regime_name = args.experiment
    regime_cfg = REGIMES[regime_name]

    print(f"Experiment: {regime_name}")

    if args.sort_parquet:
        print("Sorting parquet files...")
        sort_config = {
            "base_path": BASE_PATH,
            "baseline_subdir": regime_cfg["baseline_subdir"],
            "sgd_subdir": regime_cfg["sgd_subdir"],
        }
        sort_parquet(sort_config, key_cols=BL_KEY_COLS)
        print("Done! Re-run without --sort-parquet to analyze.")
        return

    stats = get_stats(regime_name, regime_cfg, force_recompute=args.recompute)
    generate_plots(regime_name, regime_cfg, stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
