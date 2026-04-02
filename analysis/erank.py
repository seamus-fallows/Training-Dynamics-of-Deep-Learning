"""Plot effective rank (erank) of partial products: GD (solid) vs SGD (dashed).

erank = exp(H(p)) where p_i = σ_i / Σσ_j and H is Shannon entropy.
This gives a smooth, continuous measure of how many singular values
effectively contribute to the matrix.

Figure layout mirrors sv_dynamics.py:
  - Individual layers: 4 rows (W_0 … W_3) × 3 cols (gamma)
  - Span-2 products:   3 rows × 3 cols
  - Span-3 products:   2 rows × 3 cols
  - Full product:      1 row  × 3 cols

Usage:
    python analysis/erank.py [--recompute]
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from _cache import load_fingerprinted_cache, save_fingerprinted_cache
from _common import (
    CACHE_DIR, GAMMA_NAMES,
    compute_ci, data_fingerprint, pp_label, sync_ylims,
)


# ── Config ──────────────────────────────────────────────────────────────

L = 4
SGD_BATCH_SIZE = 1

GAMMA_CONFIGS = [(0.75, 5000), (1.0, 8000), (1.5, 26000)]
GAMMA_MAX_STEPS = dict(GAMMA_CONFIGS)

INDIVIDUAL = [(i, i) for i in range(L)]
COMPOSITE_FIGURES = [
    ("span2", "Span-2 Product erank", [(0, 1), (1, 2), (2, 3)]),
    ("span3", "Span-3 Product erank", [(0, 2), (1, 3)]),
    ("full", "Full Product erank", [(0, 3)]),
]

PP_COLS = [f"pp_{i}_{j}_sv" for i in range(L) for j in range(i, L)]

# Key columns for batched parquet reads (coarsest grouping)
BATCH_KEY_COLS = ["model.hidden_dim", "model.gamma", "data.noise_std"]
ALL_KEY_COLS = BATCH_KEY_COLS + ["model.model_seed"]

DATA_ROOT = Path("outputs/sv_dynamics")
OUT_DIR = Path("figures/erank")


# ── Data loading ────────────────────────────────────────────────────────

DATA_SUBDIRS = [
    "offline/full_batch", "offline/mini_batch",
    "online/large_batch", "online/mini_batch",
]



def _parquet_path(name):
    return DATA_ROOT / name / "results.parquet"




# ── Effective rank ──────────────────────────────────────────────────────

def compute_erank(sv_array):
    """Compute effective rank from singular values.

    sv_array: (..., n_svs) — arbitrary batch dimensions.
    Returns: (...) — erank for each element.
    """
    # Normalise to probability distribution along last axis
    sv_pos = np.maximum(sv_array, 0.0)
    total = sv_pos.sum(axis=-1, keepdims=True)
    # Avoid division by zero for all-zero rows
    total = np.where(total > 0, total, 1.0)
    p = sv_pos / total

    # Shannon entropy: -Σ p_i log(p_i), treating 0·log(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    entropy = -(p * log_p).sum(axis=-1)

    return np.exp(entropy)


# ── Statistics ──────────────────────────────────────────────────────────

compute_scalar_stats = compute_ci


# ── Caching ─────────────────────────────────────────────────────────────

def _data_fingerprint():
    return data_fingerprint(DATA_ROOT, DATA_SUBDIRS)


BATCH_SIZE = 500  # rows per pyarrow batch — tested safe for ~2GB free RAM


def _compute_erank_for_subdir(subdir, batch_size_filter=None):
    """Compute per-config erank stats for one data subdirectory.

    Streams the parquet in row batches via pyarrow.iter_batches to avoid
    loading the full file into memory (~3GB). Computes erank per batch
    and accumulates only the scalar curves.

    Returns: dict[(hd, gamma, noise_std, model_seed)] -> erank_stats
    """
    path = _parquet_path(subdir)
    pf = pq.ParquetFile(path)
    schema_names = pf.schema_arrow.names

    # Determine which columns to read
    key_cols = list(ALL_KEY_COLS) + ["max_steps", "step"]
    has_bs = "training.batch_size" in schema_names
    if has_bs:
        key_cols.append("training.batch_size")
    read_cols = key_cols + PP_COLS

    # Accumulate per-config erank curves: config_key -> {col -> list of (n_seeds, n_steps) arrays}
    config_erank_curves = {}
    config_steps = {}

    n_batches = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
        n_batches += 1
        if n_batches % 20 == 0:
            print(f"    batch {n_batches}...", flush=True)
        batch_df = pl.from_arrow(batch)
        del batch

        # Gamma/max_steps filter
        gamma_mask = pl.lit(False)
        for gamma, max_steps in GAMMA_CONFIGS:
            gamma_mask = gamma_mask | (
                (pl.col("model.gamma") == gamma) & (pl.col("max_steps") == max_steps)
            )
        batch_df = batch_df.filter(gamma_mask)

        if batch_size_filter is not None and has_bs:
            batch_df = batch_df.filter(pl.col("training.batch_size") == batch_size_filter)

        if len(batch_df) == 0:
            continue

        # Group by config within this batch
        groups = batch_df.partition_by(ALL_KEY_COLS, as_dict=True)

        for group_key, group_df in groups.items():
            hd, gamma, noise_std, model_seed = group_key
            config_key = (hd, gamma, noise_std, model_seed)
            n = len(group_df)

            if config_key not in config_steps:
                config_steps[config_key] = np.array(group_df["step"][0])
                config_erank_curves[config_key] = {col: [] for col in PP_COLS}

            for col in PP_COLS:
                sv_curves = np.stack([np.stack(group_df[col][i]) for i in range(n)])
                erank_vals = compute_erank(sv_curves)  # (n_rows, n_steps)
                config_erank_curves[config_key][col].append(erank_vals)

        del batch_df, groups

    # Assemble: concatenate curves from all batches, compute stats
    result = {}
    for config_key, col_curves in config_erank_curves.items():
        stats = {"steps": config_steps[config_key]}
        for col in PP_COLS:
            if not col_curves[col]:
                continue
            all_curves = np.concatenate(col_curves[col], axis=0)
            mean, ci_lo, ci_hi = compute_scalar_stats(all_curves)
            stats[col] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}
        stats["n_seeds"] = sum(c.shape[0] for c in col_curves[PP_COLS[0]])
        result[config_key] = stats

    return result


def compute_all_stats(recompute=False):
    """Compute or load cached erank stats for all configurations.

    Streams each parquet file in row batches via pyarrow.iter_batches,
    computing erank per batch and accumulating only scalar curves.

    Returns: dict[key] -> (gd_stats, sgd_stats)
      key = (regime_key, noise_std, hd, model_seed, gamma)
    """
    cache_path = CACHE_DIR / "erank_stats.pkl"
    fingerprint = _data_fingerprint()

    if not recompute:
        cached = load_fingerprinted_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"Loaded cached erank stats ({len(cached['stats'])} configs)")
            return cached["stats"]

    all_stats = {}

    for regime_key, regime in REGIMES.items():
        # GD / large-batch baseline
        print(f"[{regime_key}] Computing GD erank ({regime['gd_data']})...")
        gd_results = _compute_erank_for_subdir(
            regime["gd_data"], batch_size_filter=regime["gd_batch_size"],
        )
        print(f"  {len(gd_results)} baseline configs")

        # SGD / mini-batch
        print(f"[{regime_key}] Computing SGD erank ({regime['sgd_data']})...")
        sgd_results = _compute_erank_for_subdir(
            regime["sgd_data"], batch_size_filter=SGD_BATCH_SIZE,
        )
        print(f"  {len(sgd_results)} SGD configs")

        # Pair up GD and SGD results
        for config_key, gd_s in gd_results.items():
            hd, gamma, noise_std, model_seed = config_key
            sgd_s = sgd_results.get(config_key)
            stat_key = (regime_key, noise_std, hd, model_seed, gamma)
            all_stats[stat_key] = (gd_s, sgd_s)

    save_fingerprinted_cache({"stats": all_stats}, cache_path, fingerprint)
    print(f"Computed and cached erank stats ({len(all_stats)} configs)")

    return all_stats


# ── Plotting ────────────────────────────────────────────────────────────

GD_COLOR = "C0"
SGD_COLOR = "C3"


def pp_ylabel(i, j):
    return f"erank({pp_label(i, j)})"


def plot_erank_panel(ax, gd, sgd, pp_col):
    """Plot GD (solid) and SGD (dashed) erank on a single axis."""
    ax.plot(gd["steps"], gd[pp_col]["mean"], color=GD_COLOR, linewidth=1.4)
    ax.plot(sgd["steps"], sgd[pp_col]["mean"], color=SGD_COLOR, linewidth=1.4,
            linestyle="--")


def make_individual_figure(gamma_gd, gamma_sgd, gd_label, sgd_label):
    """4 rows (layers) × 3 cols (gamma)."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = L
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        gd, sgd = gamma_gd[gamma], gamma_sgd[gamma]
        for r, (i, j) in enumerate(INDIVIDUAL):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_erank_panel(ax, gd, sgd, col)

            if c == 0:
                ax.set_ylabel(pp_ylabel(i, j), fontsize=10)
            if r == 0:
                ax.set_title(GAMMA_NAMES[gamma], fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color=GD_COLOR, linewidth=1.4, label=gd_label),
        Line2D([], [], color=SGD_COLOR, linewidth=1.4, linestyle="--",
               label=sgd_label),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower right")
    sync_ylims(axes)
    fig.tight_layout()
    return fig


def make_composite_figure(gamma_gd, gamma_sgd, products, gd_label, sgd_label,
                          legend_kwargs=None):
    """Variable rows (products) × 3 cols (gamma)."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = len(products)
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        gd, sgd = gamma_gd[gamma], gamma_sgd[gamma]
        for r, (i, j) in enumerate(products):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_erank_panel(ax, gd, sgd, col)

            if c == 0:
                ax.set_ylabel(pp_ylabel(i, j), fontsize=10)
            if r == 0:
                ax.set_title(GAMMA_NAMES[gamma], fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color=GD_COLOR, linewidth=1.4, label=gd_label),
        Line2D([], [], color=SGD_COLOR, linewidth=1.4, linestyle="--",
               label=sgd_label),
    ]
    leg_kw = {"fontsize": 7, "loc": "lower right"}
    if legend_kwargs:
        leg_kw.update(legend_kwargs)
    axes[0, -1].legend(handles=handles, **leg_kw)
    sync_ylims(axes)
    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────

REGIMES = {
    "offline": {
        "gd_data": "offline/full_batch",
        "sgd_data": "offline/mini_batch",
        "label": "Offline (N=500)",
        "gd_label": "GD (full batch)",
        "sgd_label": f"SGD (B={SGD_BATCH_SIZE})",
        "gd_batch_size": None,
    },
    "online": {
        "gd_data": "online/large_batch",
        "sgd_data": "online/mini_batch",
        "label": "Online (infinite data)",
        "gd_label": "Large batch",
        "sgd_label": f"Mini-batch (B={SGD_BATCH_SIZE})",
        "gd_batch_size": None,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true",
                        help="Force recompute statistics (ignore cache)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    all_stats = compute_all_stats(recompute=args.recompute)
    t_stats = time.perf_counter() - t0
    print(f"Stats ready in {t_stats:.1f}s")

    hidden_dims = sorted(set(k[2] for k in all_stats))
    noise_stds = sorted(set(k[1] for k in all_stats))
    model_seeds = sorted(set(k[3] for k in all_stats))

    jobs = []
    for regime_key, regime in REGIMES.items():
        for noise_std in noise_stds:
            fig_dir = OUT_DIR / regime_key / f"noise_{noise_std}"
            fig_dir.mkdir(parents=True, exist_ok=True)

            for hd in hidden_dims:
                for model_seed in model_seeds:
                    skip = False
                    for gamma, _ in GAMMA_CONFIGS:
                        key = (regime_key, noise_std, hd, model_seed, gamma)
                        gd, sgd = all_stats[key]
                        if gd is None or sgd is None:
                            print(f"  SKIP {regime_key} hd={hd} seed={model_seed} "
                                  f"g={gamma} noise={noise_std}: missing data")
                            skip = True
                            break
                    if skip:
                        continue

                    base = (regime_key, noise_std, hd, model_seed,
                            regime["gd_label"], regime["sgd_label"])

                    jobs.append((
                        "individual", *base,
                        fig_dir / f"individual_h{hd}_seed{model_seed}.png",
                        {},
                    ))

                    for slug, _, products in COMPOSITE_FIGURES:
                        extra = {"products": products}
                        jobs.append((
                            "composite", *base,
                            fig_dir / f"{slug}_h{hd}_seed{model_seed}.png",
                            extra,
                        ))

    t0 = time.perf_counter()
    print(f"Rendering {len(jobs)} figures...")

    for job in jobs:
        fig_type, regime_key, noise_std, hd, model_seed, gd_label, sgd_label, path, extra = job

        gamma_gd, gamma_sgd = {}, {}
        for gamma, _ in GAMMA_CONFIGS:
            key = (regime_key, noise_std, hd, model_seed, gamma)
            gd, sgd = all_stats[key]
            gamma_gd[gamma] = gd
            gamma_sgd[gamma] = sgd

        if fig_type == "individual":
            fig = make_individual_figure(gamma_gd, gamma_sgd, gd_label, sgd_label)
        else:
            fig = make_composite_figure(gamma_gd, gamma_sgd, extra["products"],
                                        gd_label, sgd_label,
                                        legend_kwargs=extra.get("legend_kwargs"))

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    t_render = time.perf_counter() - t0
    print(f"Rendered {len(jobs)} figures in {t_render:.1f}s")
    print(f"\nDone. Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
