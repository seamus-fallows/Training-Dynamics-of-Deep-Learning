"""Plot partial-product SV dynamics: GD (solid) vs SGD (dashed, mean ± 99% CI).

Three figure types per (hidden_dim, model_seed, regime, noise_std):

  - Individual layers: 4 rows (W_0 … W_3) × 3 cols (gamma)
  - Span-2 products:   3 rows (W_1·W_0,  W_2·W_1,  W_3·W_2) × 3 cols (gamma)
  - Span-3:            2 rows (W_2·W_1·W_0,  W_3·W_2·W_1) × 3 cols (gamma)
  - Full product:      1 row  (W_3···W_0) × 3 cols (gamma)

Additionally generates percentage-difference figures (SGD vs GD) in a separate folder.

SGD curves are averaged over batch seeds with shaded ± 1σ envelopes.

Usage:
    python analysis/sv_dynamics.py [--recompute]
"""

import argparse
import hashlib
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import polars as pl

from _common import CACHE_DIR, GAMMA_NAMES


# ── Config ──────────────────────────────────────────────────────────────

L = 4  # 3 hidden layers → 4 weight matrices
N_SV_PLOT = 10
SGD_BATCH_SIZE = 1

GAMMA_CONFIGS = [(0.75, 5000), (1.0, 8000), (1.5, 26000)]

# Partial product groupings
INDIVIDUAL = [(i, i) for i in range(L)]
COMPOSITE_FIGURES = [
    ("span2", "Span-2 Product SVs", [(0, 1), (1, 2), (2, 3)]),
    ("span3", "Span-3 Product SVs", [(0, 2), (1, 3)]),
    ("full", "Full Product SVs", [(0, 3)]),
]

PP_COLS = [f"pp_{i}_{j}_sv" for i in range(L) for j in range(i, L)]

DATA_ROOT = Path("outputs/sv_dynamics")
OUT_DIR = Path("figures/sv_dynamics")
PCTDIFF_OUT_DIR = Path("figures/sv_dynamics_pctdiff")
SV_COLORS = [f"C{i}" for i in range(N_SV_PLOT)]


# ── Data loading ────────────────────────────────────────────────────────

DATA_SUBDIRS = [
    "offline/full_batch", "offline/mini_batch",
    "online/large_batch", "online/mini_batch",
]


def _parquet_path(name):
    return DATA_ROOT / name / "results.parquet"


def _config_filter(gamma, max_steps, hd, model_seed, noise_std, batch_size=None):
    filt = (
        (pl.col("model.gamma") == gamma)
        & (pl.col("max_steps") == max_steps)
        & (pl.col("model.hidden_dim") == hd)
        & (pl.col("model.model_seed") == model_seed)
        & (pl.col("data.noise_std") == noise_std)
    )
    if batch_size is not None:
        filt = filt & (pl.col("training.batch_size") == batch_size)
    return filt


def extract_sv_array(df, col, n_rows):
    """Extract SV data from a polars list-of-lists column -> (n_rows, n_steps, n_svs)."""
    return np.stack([np.stack(df[col][i]) for i in range(n_rows)])


# ── Statistics ──────────────────────────────────────────────────────────

def compute_sv_stats(curves):
    """Compute mean and 95% CI for SV curves.

    curves: (n_seeds, n_steps, n_svs)
    Returns: mean, ci_lo, ci_hi — each (n_steps, n_svs)
    """
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    if n == 1:
        return mean, mean, mean

    sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
    # 95% CI: z ≈ 1.96 for large n
    z = 1.96
    return mean, mean - z * sem, mean + z * sem


def _filter_rows(df, gamma, max_steps, hd, model_seed, noise_std, batch_size=None):
    """Filter an in-memory DataFrame to one config."""
    filt = _config_filter(gamma, max_steps, hd, model_seed, noise_std, batch_size)
    return df.filter(filt)


def _rows_to_stats(rows):
    """Compute SV stats from filtered rows -> dict with steps + per-col stats."""
    n = len(rows)
    if n == 0:
        return None
    steps = np.array(rows["step"][0])
    result = {"steps": steps, "n_seeds": n}
    for col in PP_COLS:
        curves = extract_sv_array(rows, col, n)[:, :, :N_SV_PLOT]
        mean, ci_lo, ci_hi = compute_sv_stats(curves)
        result[col] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}
    return result


def _compute_pct_diff_from_arrays(gd_arrays, sgd_arrays, steps_gd, steps_sgd):
    """Compute per-SV percentage difference from pre-extracted numpy arrays.

    gd_arrays/sgd_arrays: dict[col] -> (n_seeds, n_steps, n_svs)
    """
    result = {"steps": steps_sgd}

    for col in PP_COLS:
        if col not in gd_arrays or col not in sgd_arrays:
            continue
        gd_curves = gd_arrays[col]
        sgd_curves = sgd_arrays[col]
        gd_mean = gd_curves.mean(axis=0)
        n_svs = min(gd_mean.shape[1], sgd_curves.shape[2])

        if np.array_equal(steps_gd, steps_sgd):
            gd_ref = gd_mean[:, :n_svs]
        else:
            gd_ref = np.column_stack([
                np.interp(steps_sgd, steps_gd, gd_mean[:, k])
                for k in range(n_svs)
            ])

        ref = np.where(np.abs(gd_ref) > 1e-10, gd_ref, np.nan)
        pct_curves = (sgd_curves[:, :, :n_svs] - gd_ref[None]) / np.abs(ref[None]) * 100
        mean, ci_lo, ci_hi = compute_sv_stats(pct_curves)
        result[col] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}

    return result


def _process_config(args):
    """Worker function for parallel stats computation.

    Takes pre-extracted numpy arrays (not polars DataFrames) so it can be
    pickled for ProcessPoolExecutor.
    """
    key, gd_arrays, sgd_arrays, gd_steps, sgd_steps, gd_n, sgd_n, need_stats, need_pctdiff = args

    stats_result = None
    pctdiff_result = None

    if need_stats:
        # GD stats
        if gd_n == 0:
            gd_stats = None
        else:
            gd_stats = {"steps": gd_steps, "n_seeds": gd_n}
            for col, curves in gd_arrays.items():
                mean, ci_lo, ci_hi = compute_sv_stats(curves)
                gd_stats[col] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}

        # SGD stats
        if sgd_n == 0:
            sgd_stats = None
        else:
            sgd_stats = {"steps": sgd_steps, "n_seeds": sgd_n}
            for col, curves in sgd_arrays.items():
                mean, ci_lo, ci_hi = compute_sv_stats(curves)
                sgd_stats[col] = {"mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi}

        stats_result = (gd_stats, sgd_stats)

    if need_pctdiff:
        if gd_n == 0 or sgd_n == 0:
            pctdiff_result = None
        else:
            pctdiff_result = _compute_pct_diff_from_arrays(
                gd_arrays, sgd_arrays, gd_steps, sgd_steps,
            )

    return key, stats_result, pctdiff_result


# ── Caching ─────────────────────────────────────────────────────────────

def _data_fingerprint():
    """Hash based on mtime+size of all parquet files — invalidates on data change."""
    h = hashlib.md5()
    for name in DATA_SUBDIRS:
        p = DATA_ROOT / name / "results.parquet"
        if p.exists():
            stat = p.stat()
            h.update(f"{p}:{stat.st_mtime}:{stat.st_size}".encode())
    return h.hexdigest()[:12]


def compute_all_stats(recompute=False):
    """Compute or load cached stats + pct-diff for all configurations.

    Single pass: loads each config's GD/SGD rows once, computes both regular stats
    and pct-diff stats, then frees the rows. Memory stays bounded.

    Returns: (all_stats, all_pct_diff) where
      all_stats:    dict[key] -> (gd_stats, sgd_stats)
      all_pct_diff: dict[key] -> pct_diff_stats
      key = (regime_key, noise_std, hd, model_seed, gamma)
    """
    stats_cache = CACHE_DIR / "sv_dynamics_stats.pkl"
    pctdiff_cache = CACHE_DIR / "sv_dynamics_pctdiff_stats.pkl"
    fingerprint = _data_fingerprint()

    # Try loading both caches
    stats_ok, pctdiff_ok = False, False
    all_stats, all_pct_diff = None, None

    if not recompute:
        if stats_cache.exists():
            with open(stats_cache, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                all_stats = cached["stats"]
                stats_ok = True
                print(f"Loaded cached stats ({len(all_stats)} configs)")

        if pctdiff_cache.exists():
            with open(pctdiff_cache, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == fingerprint:
                all_pct_diff = cached["stats"]
                pctdiff_ok = True
                print(f"Loaded cached pct-diff stats ({len(all_pct_diff)} configs)")

    if stats_ok and pctdiff_ok:
        return all_stats, all_pct_diff

    # Discover sweep parameters
    df_ref = pl.read_parquet(
        _parquet_path("offline/full_batch"),
        columns=["model.hidden_dim", "data.noise_std", "model.model_seed"],
    )
    hidden_dims = sorted(df_ref["model.hidden_dim"].unique().to_list())
    noise_stds = sorted(df_ref["data.noise_std"].unique().to_list())
    model_seeds = sorted(df_ref["model.model_seed"].unique().to_list())
    del df_ref

    need_stats = not stats_ok
    need_pctdiff = not pctdiff_ok
    if need_stats:
        all_stats = {}
    if need_pctdiff:
        all_pct_diff = {}

    n_configs = (len(REGIMES) * len(noise_stds) * len(hidden_dims)
                 * len(model_seeds) * len(GAMMA_CONFIGS))

    n_workers = min(os.cpu_count() or 4, 12)

    for regime_key, regime in REGIMES.items():
        # Load each parquet file once — all subsequent filters are in-memory
        print(f"Loading {regime_key} parquet files...")
        gd_df = pl.read_parquet(_parquet_path(regime["gd_data"]))
        sgd_df = pl.read_parquet(_parquet_path(regime["sgd_data"]))

        # Extract numpy arrays in main thread (fast), submit computation to workers
        work_items = []
        for noise_std in noise_stds:
            for hd in hidden_dims:
                for model_seed in model_seeds:
                    for gamma, max_steps in GAMMA_CONFIGS:
                        key = (regime_key, noise_std, hd, model_seed, gamma)

                        gd_rows = _filter_rows(
                            gd_df, gamma, max_steps, hd, model_seed, noise_std,
                            batch_size=regime["gd_batch_size"],
                        )
                        sgd_rows = _filter_rows(
                            sgd_df, gamma, max_steps, hd, model_seed, noise_std,
                            batch_size=SGD_BATCH_SIZE,
                        )

                        # Pre-extract numpy arrays so workers don't need polars
                        gd_n, sgd_n = len(gd_rows), len(sgd_rows)
                        gd_steps = np.array(gd_rows["step"][0]) if gd_n > 0 else None
                        sgd_steps = np.array(sgd_rows["step"][0]) if sgd_n > 0 else None
                        gd_arrays = {col: extract_sv_array(gd_rows, col, gd_n)[:, :, :N_SV_PLOT]
                                     for col in PP_COLS} if gd_n > 0 else {}
                        sgd_arrays = {col: extract_sv_array(sgd_rows, col, sgd_n)[:, :, :N_SV_PLOT]
                                      for col in PP_COLS} if sgd_n > 0 else {}

                        work_items.append((
                            key, gd_arrays, sgd_arrays, gd_steps, sgd_steps,
                            gd_n, sgd_n, need_stats, need_pctdiff,
                        ))

        del gd_df, sgd_df

        print(f"  Processing {len(work_items)} configs with {n_workers} threads...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, (key, stats_result, pctdiff_result) in enumerate(
                pool.map(_process_config, work_items), 1,
            ):
                if need_stats:
                    all_stats[key] = stats_result
                if need_pctdiff:
                    all_pct_diff[key] = pctdiff_result
                if i % 12 == 0:
                    print(f"  {i}/{len(work_items)} configs done...")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if need_stats:
        with open(stats_cache, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "stats": all_stats}, f)
        print(f"Computed and cached stats ({len(all_stats)} configs)")
    if need_pctdiff:
        with open(pctdiff_cache, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "stats": all_pct_diff}, f)
        print(f"Computed and cached pct-diff stats ({len(all_pct_diff)} configs)")

    return all_stats, all_pct_diff


# ── Plotting ────────────────────────────────────────────────────────────

def pp_label(i, j):
    if i == j:
        return f"$W_{{{j}}}$"
    layers = " ".join(f"W_{{{k}}}" for k in range(j, i - 1, -1))
    return f"${layers}$"


def pp_ylabel(i, j, n_total_svs=None):
    label = pp_label(i, j)
    if n_total_svs is not None and n_total_svs >= N_SV_PLOT:
        return f"Top {N_SV_PLOT} SVs of {label}"
    return f"SVs of {label}"


def gamma_label(gamma):
    return GAMMA_NAMES[gamma]


def plot_sv_panel(ax, gd, sgd, pp_col):
    """Plot GD (solid) and SGD (dashed) on a single axis."""
    steps_gd = gd["steps"]
    gd_sv = gd[pp_col]
    n_plot = min(N_SV_PLOT, gd_sv["mean"].shape[1])
    for k in range(n_plot):
        ax.plot(steps_gd, gd_sv["mean"][:, k], color=SV_COLORS[k],
                linewidth=1.2)

    steps_sgd = sgd["steps"]
    sgd_sv = sgd[pp_col]
    for k in range(n_plot):
        ax.plot(steps_sgd, sgd_sv["mean"][:, k], color=SV_COLORS[k],
                linewidth=1.2, linestyle="--")


def make_individual_figure(gamma_gd, gamma_sgd, gd_label, sgd_label):
    """4 rows (layers) x 3 cols (gamma)."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = L
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        gd, sgd = gamma_gd[gamma], gamma_sgd[gamma]

        for r, (i, j) in enumerate(INDIVIDUAL):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_sv_panel(ax, gd, sgd, col)

            if c == 0:
                n_total = gd[col]["mean"].shape[1]
                ax.set_ylabel(pp_ylabel(i, j, n_total), fontsize=10)
            if r == 0:
                ax.set_title(gamma_label(gamma), fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color="gray", linewidth=1.2, label=gd_label),
        Line2D([], [], color="gray", linewidth=1.2, linestyle="--",
               label=sgd_label),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    return fig


def make_composite_figure(gamma_gd, gamma_sgd, products, gd_label, sgd_label,
                          legend_kwargs=None):
    """3 rows (products) x 3 cols (gamma)."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = len(products)
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        gd, sgd = gamma_gd[gamma], gamma_sgd[gamma]

        for r, (i, j) in enumerate(products):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_sv_panel(ax, gd, sgd, col)

            if c == 0:
                n_total = gd[col]["mean"].shape[1]
                ax.set_ylabel(pp_ylabel(i, j, n_total), fontsize=10)
            if r == 0:
                ax.set_title(gamma_label(gamma), fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color="gray", linewidth=1.2, label=gd_label),
        Line2D([], [], color="gray", linewidth=1.2, linestyle="--",
               label=sgd_label),
    ]
    leg_kw = {"fontsize": 7, "loc": "lower right"}
    if legend_kwargs:
        leg_kw.update(legend_kwargs)
    axes[0, -1].legend(handles=handles, **leg_kw)

    fig.tight_layout()
    return fig


# ── Percentage-difference plotting ─────────────────────────────────────

def pp_ylabel_pctdiff(i, j, n_total_svs=None):
    label = pp_label(i, j)
    if n_total_svs is not None and n_total_svs >= N_SV_PLOT:
        return f"% Diff Top {N_SV_PLOT} SVs of {label}"
    return f"% Diff SVs of {label}"


def plot_pctdiff_panel(ax, pct_diff, pp_col):
    """Plot percentage difference per SV with 99% CI shading."""
    steps = pct_diff["steps"]
    sv_data = pct_diff[pp_col]
    n_plot = sv_data["mean"].shape[1]
    for k in range(n_plot):
        ax.plot(steps, sv_data["mean"][:, k], color=SV_COLORS[k], linewidth=1.2)
        if sv_data["ci_lo"] is not sv_data["mean"]:
            ax.fill_between(steps, sv_data["ci_lo"][:, k], sv_data["ci_hi"][:, k],
                            color=SV_COLORS[k], alpha=0.15)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")


def make_individual_pctdiff_figure(gamma_pctdiff):
    """4 rows (layers) × 3 cols (gamma) — percentage difference."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = L
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        pct = gamma_pctdiff[gamma]

        for r, (i, j) in enumerate(INDIVIDUAL):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_pctdiff_panel(ax, pct, col)

            if c == 0:
                n_total = pct[col]["mean"].shape[1]
                ax.set_ylabel(pp_ylabel_pctdiff(i, j, n_total), fontsize=10)
            if r == 0:
                ax.set_title(gamma_label(gamma), fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color="gray", linewidth=1.2, label="(SGD − GD) / |GD| (95% CI)"),
    ]
    axes[0, -1].legend(handles=handles, fontsize=7, loc="lower right")

    fig.tight_layout()
    return fig


def make_composite_pctdiff_figure(gamma_pctdiff, products, legend_kwargs=None):
    """3 rows (products) × 3 cols (gamma) — percentage difference."""
    n_gammas = len(GAMMA_CONFIGS)
    n_rows = len(products)
    fig, axes = plt.subplots(n_rows, n_gammas, figsize=(4.2 * n_gammas, 3.0 * n_rows),
                             squeeze=False)

    for c, (gamma, _) in enumerate(GAMMA_CONFIGS):
        pct = gamma_pctdiff[gamma]

        for r, (i, j) in enumerate(products):
            ax = axes[r, c]
            col = f"pp_{i}_{j}_sv"
            plot_pctdiff_panel(ax, pct, col)

            if c == 0:
                n_total = pct[col]["mean"].shape[1]
                ax.set_ylabel(pp_ylabel_pctdiff(i, j, n_total), fontsize=10)
            if r == 0:
                ax.set_title(gamma_label(gamma), fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel("Step")

    handles = [
        Line2D([], [], color="gray", linewidth=1.2, label="(SGD − GD) / |GD| (95% CI)"),
    ]
    leg_kw = {"fontsize": 7, "loc": "lower right"}
    if legend_kwargs:
        leg_kw.update(legend_kwargs)
    axes[0, -1].legend(handles=handles, **leg_kw)

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
    all_stats, all_pct_diff = compute_all_stats(recompute=args.recompute)
    t_stats = time.perf_counter() - t0
    print(f"Stats ready in {t_stats:.1f}s")

    hidden_dims = sorted(set(k[2] for k in all_stats))
    noise_stds = sorted(set(k[1] for k in all_stats))
    model_seeds = sorted(set(k[3] for k in all_stats))

    # Build list of render jobs
    jobs = []
    pctdiff_jobs = []
    for regime_key, regime in REGIMES.items():
        for noise_std in noise_stds:
            fig_dir = OUT_DIR / regime_key / f"noise_{noise_std}"
            fig_dir.mkdir(parents=True, exist_ok=True)
            pctdiff_dir = PCTDIFF_OUT_DIR / regime_key / f"noise_{noise_std}"
            pctdiff_dir.mkdir(parents=True, exist_ok=True)

            for hd in hidden_dims:
                for model_seed in model_seeds:
                    # Check all gammas are present
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
                    pctdiff_jobs.append((
                        "individual", regime_key, noise_std, hd, model_seed,
                        pctdiff_dir / f"individual_h{hd}_seed{model_seed}.png",
                        {},
                    ))

                    for slug, _, products in COMPOSITE_FIGURES:
                        extra = {"products": products}
                        if slug == "full":
                            extra["legend_kwargs"] = {
                                "loc": "right",
                                "bbox_to_anchor": (1.0, 0.75),
                            }
                        jobs.append((
                            "composite", *base,
                            fig_dir / f"{slug}_h{hd}_seed{model_seed}.png",
                            extra,
                        ))
                        pctdiff_jobs.append((
                            "composite", regime_key, noise_std, hd, model_seed,
                            pctdiff_dir / f"{slug}_h{hd}_seed{model_seed}.png",
                            extra,
                        ))

    # Render sequentially — avoids memory copies from multiprocessing
    t0 = time.perf_counter()
    total = len(jobs) + len(pctdiff_jobs)
    print(f"Rendering {total} figures ({len(jobs)} SV + {len(pctdiff_jobs)} pct-diff)...")

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

    for job in pctdiff_jobs:
        fig_type, regime_key, noise_std, hd, model_seed, path, extra = job

        gamma_pctdiff = {}
        for gamma, _ in GAMMA_CONFIGS:
            key = (regime_key, noise_std, hd, model_seed, gamma)
            gamma_pctdiff[gamma] = all_pct_diff[key]

        if fig_type == "individual":
            fig = make_individual_pctdiff_figure(gamma_pctdiff)
        else:
            fig = make_composite_pctdiff_figure(gamma_pctdiff, extra["products"],
                                                legend_kwargs=extra.get("legend_kwargs"))

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    t_render = time.perf_counter() - t0
    print(f"Rendered {total} figures in {t_render:.1f}s")

    print(f"\nDone. {len(jobs)} SV figures saved to {OUT_DIR}/")
    print(f"      {len(pctdiff_jobs)} pct-diff figures saved to {PCTDIFF_OUT_DIR}/")


if __name__ == "__main__":
    main()
