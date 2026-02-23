"""
GPH Comparative Metrics Analysis

Plots comparative metrics and model metrics between a baseline model (A) and
comparison models (B), with model B quantities averaged over batch seeds.

Supports two modes:
  - offline: A = full-batch GD (deterministic), B = mini-batch SGD
  - online:  A = large-batch B=500 (fixed seed), B = mini-batch

Two figure types per (gamma, noise, width, batch_size) configuration:

  - Distances: loss context, relative param/frobenius/layer distances, cosine similarity
  - Model Metrics: loss context, layer norms, gram norms, balance diffs/ratio,
    effective weight norm  (A solid vs B dashed + 95% CI)

Data sources:
  - Comparative sweep (--input): test_loss_a/b, param_distance, layer_distances,
    frobenius_distance, plus model B metrics as _b-suffixed columns (via
    metrics_a=[] so only model_b metrics are tracked).
  - Baseline sweep (--baseline-input): layer_norms, gram_norms, balance_diffs,
    end_to_end_weight_norm for the baseline (model A) side.

Usage (run from the project root):

    python analysis/gph_comparative_metrics.py --recompute
    python analysis/gph_comparative_metrics.py --mode=online --recompute

Options:
    --mode MODE        offline (default) or online
    --input PATH       Override comparative input dir
    --baseline-input PATH  Override baseline metrics input dir
    --output PATH      Override output figures dir
    --cache PATH       Override cache file path
    --recompute        Force recompute statistics (ignore cache)
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from _common import CACHE_DIR, GAMMA_NAMES, fmt_seeds


# =============================================================================
# Configuration
# =============================================================================

MODE_CONFIGS = {
    "offline": {
        "input": Path("outputs/gph_comparative_metrics/comparative"),
        "baseline_input": Path("outputs/gph_comparative_metrics/gd_metrics"),
        "output": Path("figures/gph_comparative_metrics"),
        "cache": CACHE_DIR / "gph_comparative_metrics.pkl",
        # Labels
        "a_label": "GD",
        "b_label_fmt": "SGD (B={bs})",  # formatted with batch_size
        "b_short": "SGD",               # for per-layer legends
        "sub_a": "GD",                  # LaTeX subscripts
        "sub_b": "SGD",
    },
    "online": {
        "input": Path("outputs/gph_online_comparative_metrics/comparative"),
        "baseline_input": Path("outputs/gph_online_comparative_metrics/large_batch_metrics"),
        "output": Path("figures/gph_online_comparative_metrics"),
        "cache": CACHE_DIR / "gph_online_comparative_metrics.pkl",
        # Labels
        "a_label": "B=500",
        "b_label_fmt": "B={bs}",
        "b_short": "B={bs}",
        "sub_a": "A",
        "sub_b": "B",
    },
}

# Columns that define a unique configuration (batch_seed is averaged over)
GROUP_COLS = [
    "model.hidden_dim", "model.gamma", "data.noise_std",
    "model.model_seed", "training_b.batch_size",
]


def _axis_labels(sub_a: str, sub_b: str) -> dict[str, str]:
    """Build parametrized LaTeX axis labels for distance/cosine plots."""
    A = r"\mathrm{" + sub_a + "}"
    B = r"\mathrm{" + sub_b + "}"
    thA, thB = r"\theta_{" + A + "}", r"\theta_{" + B + "}"
    wA = r"W_{\mathrm{eff}}^{" + A + "}"
    wB = r"W_{\mathrm{eff}}^{" + B + "}"
    wiA, wiB = r"W_i^{" + A + "}", r"W_i^{" + B + "}"
    return {
        "param_rel":  r"$\|" + thB + " - " + thA + r"\| / \|" + thA + r"\|$",
        "param_abs":  r"$\|" + thB + " - " + thA + r"\|$",
        "eff_rel":    r"$\|" + wB + " - " + wA + r"\|_F / \|" + wA + r"\|_F$",
        "eff_abs":    r"$\|" + wB + " - " + wA + r"\|_F$",
        "cosine":     r"$\cos(" + thA + ", " + thB + ")$",
        "layer_rel":  r"$\|" + wiB + " - " + wiA + r"\|_F / \|" + wiA + r"\|_F$",
        "layer_abs":  r"$\|" + wiB + " - " + wiA + r"\|_F$",
    }


def _b_label(mcfg: dict, batch_size: int) -> str:
    """Full comparison label, e.g. 'SGD (B=10)' or 'B=10'."""
    return mcfg["b_label_fmt"].format(bs=batch_size)


def _b_short(mcfg: dict, batch_size: int) -> str:
    """Short comparison label for per-layer legends, e.g. 'SGD' or 'B=10'."""
    return mcfg["b_short"].format(bs=batch_size)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class MetricStats:
    """Per-metric statistics with precomputed plotting quantities."""

    mean: np.ndarray       # (n_steps,)
    n: int
    ci_lo: np.ndarray      # 95% CI lower
    ci_hi: np.ndarray      # 95% CI upper
    min_vals: np.ndarray | None   # None when n == 1
    max_vals: np.ndarray | None


@dataclass
class CompConfigStats:
    """Statistics for one (width, gamma, noise, model_seed, batch_size) group.

    Baseline (A) quantities are deterministic and stored as plain arrays.
    Comparison (B) quantities are averaged over batch seeds and stored as MetricStats.
    """

    steps: np.ndarray

    # Loss
    loss_gd: np.ndarray
    loss_sgd: MetricStats

    # Comparative distances
    param_distance: MetricStats
    frobenius_distance: MetricStats
    layer_distances: dict[int, MetricStats]

    # Cosine similarity — None when B layer norms are unavailable
    cosine_sim: MetricStats | None

    # Baseline (A) model metrics (from separate baseline sweep)
    gd_param_norm: np.ndarray | None
    gd_end_to_end_weight_norm: np.ndarray | None
    gd_layer_norms: dict[int, np.ndarray]
    gd_gram_norms: dict[int, np.ndarray]
    gd_balance_diffs: dict[int, np.ndarray]
    gd_balance_ratios: dict[int, np.ndarray]

    # Comparison (B) model metrics (from _b columns)
    sgd_layer_norms: dict[int, MetricStats]
    sgd_gram_norms: dict[int, MetricStats]
    sgd_balance_diffs: dict[int, MetricStats]
    sgd_balance_ratios: dict[int, MetricStats]
    sgd_end_to_end_weight_norm: MetricStats | None


# =============================================================================
# Statistics Helpers
# =============================================================================


def _extract_curves(df: pl.DataFrame, col: str) -> np.ndarray:
    """Extract metric curves as a (n_runs, n_steps) numpy array."""
    return np.vstack(df[col].to_list())


def _make_stats(curves: np.ndarray) -> MetricStats:
    """Build MetricStats from a (n_runs, n_steps) array."""
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1, ci_lo=mean, ci_hi=mean,
            min_vals=None, max_vals=None,
        )

    var = curves.var(axis=0, ddof=1)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(var / n)

    # Log-space CI (delta method on log(X̄))
    relative_sem = sem / np.maximum(np.abs(mean), 1e-30)
    ci_factor = np.exp(np.minimum(t_val * relative_sem, 700))

    return MetricStats(
        mean=mean, n=n,
        ci_lo=mean / ci_factor,
        ci_hi=mean * ci_factor,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
    )


def _make_stats_linear_ci(curves: np.ndarray) -> MetricStats:
    """Build MetricStats with linear (not log-space) 95% CI.

    Appropriate for metrics that can be negative or near-constant,
    like cosine similarity or balance ratio.
    """
    n = len(curves)
    mean = curves.mean(axis=0)

    if n == 1:
        return MetricStats(
            mean=mean, n=1, ci_lo=mean, ci_hi=mean,
            min_vals=None, max_vals=None,
        )

    var = curves.var(axis=0, ddof=1)
    t_val = scipy_stats.t.ppf(0.975, df=n - 1)
    sem = np.sqrt(var / n)

    return MetricStats(
        mean=mean, n=n,
        ci_lo=mean - t_val * sem,
        ci_hi=mean + t_val * sem,
        min_vals=curves.min(axis=0),
        max_vals=curves.max(axis=0),
    )


# =============================================================================
# Baseline Fallback
# =============================================================================


def _load_baseline_fallback(
    baseline_dir: Path,
) -> dict[tuple, dict]:
    """Load baseline model metrics for normalisation and model_metrics figure.

    Returns a dict keyed by (width, gamma, noise, seed) with layer norms,
    gram norms, balance diffs/ratios, param norm, and effective weight norm.
    """
    path = baseline_dir / "results.parquet"
    if not path.exists():
        print(f"  Baseline fallback not found at {path}")
        return {}

    print(f"  Loading baseline fallback from {path}...")
    df = pl.read_parquet(path)

    # Detect layers
    n_layers = 0
    while f"layer_norm_{n_layers}" in df.columns:
        n_layers += 1
    n_pairs = max(n_layers - 1, 0)

    result = {}
    for row in df.iter_rows(named=True):
        key = (
            row["model.hidden_dim"], row["model.gamma"],
            row["data.noise_std"], row["model.model_seed"],
        )

        layer_norms = {
            i: np.array(row[f"layer_norm_{i}"])
            for i in range(n_layers)
            if f"layer_norm_{i}" in row and row[f"layer_norm_{i}"] is not None
        }
        gram_norms = {
            i: np.array(row[f"gram_norm_{i}"])
            for i in range(n_layers)
            if f"gram_norm_{i}" in row and row[f"gram_norm_{i}"] is not None
        }
        balance_diffs = {}
        balance_ratios = {}
        for i in range(n_pairs):
            col = f"balance_diff_{i}"
            if col in row and row[col] is not None:
                balance_diffs[i] = np.array(row[col])
                if i in gram_norms and (i + 1) in gram_norms:
                    denom = gram_norms[i] + gram_norms[i + 1]
                    balance_ratios[i] = balance_diffs[i] / np.maximum(denom, 1e-30)

        param_norm = None
        if layer_norms:
            param_norm = np.sqrt(sum(v ** 2 for v in layer_norms.values()))

        eff = None
        if "end_to_end_weight_norm" in row and row["end_to_end_weight_norm"] is not None:
            eff = np.array(row["end_to_end_weight_norm"])

        result[key] = {
            "layer_norms": layer_norms,
            "gram_norms": gram_norms,
            "balance_diffs": balance_diffs,
            "balance_ratios": balance_ratios,
            "param_norm": param_norm,
            "end_to_end_weight_norm": eff,
        }

    print(f"  {len(result)} baseline configs loaded as fallback")
    return result


# =============================================================================
# Data Loading & Stats Computation
# =============================================================================


def _detect_comp_layers(df: pl.DataFrame) -> int:
    i = 0
    while f"layer_distance_{i}" in df.columns:
        i += 1
    return i


def _detect_sgd_model_layers(df: pl.DataFrame) -> int:
    """Detect _b suffixed model layers in comparative data."""
    i = 0
    while f"layer_norm_{i}_b" in df.columns:
        i += 1
    return i


def compute_all_stats(
    input_dir: Path,
    baseline_fallback: dict,
) -> dict[tuple, CompConfigStats]:
    """Load comparative results and compute per-config stats over batch seeds.

    Model B metrics are sourced from _b columns in comparative data (via
    metrics_a=[] so only model_b metrics are tracked).
    Baseline (A) model metrics always come from baseline_fallback.
    """
    path = input_dir / "results.parquet"
    print(f"Loading {path}...")
    df = pl.read_parquet(path)
    print(f"  {len(df)} runs loaded")

    n_comp_layers = _detect_comp_layers(df)
    n_sgd_layers = _detect_sgd_model_layers(df)
    has_sgd_metrics = n_sgd_layers > 0
    n_sgd_pairs = max(n_sgd_layers - 1, 0)

    print(f"  {n_comp_layers} layer distance cols")
    if has_sgd_metrics:
        print(f"  {n_sgd_layers} model B layers (_b columns)")
    else:
        print(f"  No model B metrics (re-run comparative sweep with model metrics enabled)")

    groups = df.partition_by(GROUP_COLS, as_dict=True)
    total = len(groups)
    print(f"  Computing stats for {total} config groups...")

    stats: dict[tuple, CompConfigStats] = {}
    for idx, (key, gdf) in enumerate(groups.items()):
        if len(gdf) == 0:
            continue

        steps = np.array(gdf["step"][0])
        width, gamma, noise, seed, batch_size = key

        # -- Loss --
        loss_gd = _extract_curves(gdf, "test_loss_a").mean(axis=0)
        loss_sgd = _make_stats(_extract_curves(gdf, "test_loss_b"))

        # -- Comparative distances --
        param_dist_curves = _extract_curves(gdf, "param_distance")
        param_distance = _make_stats(param_dist_curves)
        frobenius_distance = _make_stats(_extract_curves(gdf, "frobenius_distance"))

        layer_distances: dict[int, MetricStats] = {}
        for i in range(n_comp_layers):
            layer_distances[i] = _make_stats(_extract_curves(gdf, f"layer_distance_{i}"))

        # -- Baseline (A) model metrics (always from fallback) --
        bl_key = (width, gamma, noise, seed)
        bl = baseline_fallback.get(bl_key, {})
        gd_layer_norms = bl.get("layer_norms", {})
        gd_gram_norms = bl.get("gram_norms", {})
        gd_balance_diffs = bl.get("balance_diffs", {})
        gd_balance_ratios = bl.get("balance_ratios", {})
        gd_param_norm = bl.get("param_norm")
        gd_end_to_end_weight_norm = bl.get("end_to_end_weight_norm")

        # -- Model B metrics & cosine sim --
        sgd_layer_norms: dict[int, MetricStats] = {}
        sgd_gram_norms: dict[int, MetricStats] = {}
        sgd_balance_diffs_out: dict[int, MetricStats] = {}
        sgd_balance_ratios_out: dict[int, MetricStats] = {}
        sgd_end_to_end_weight_norm: MetricStats | None = None
        cosine_sim: MetricStats | None = None

        if has_sgd_metrics:
            # B metrics from _b columns in comparative data
            sgd_ln_curves: dict[int, np.ndarray] = {}
            for i in range(n_sgd_layers):
                sgd_ln_curves[i] = _extract_curves(gdf, f"layer_norm_{i}_b")
                sgd_layer_norms[i] = _make_stats(sgd_ln_curves[i])

            for i in range(n_sgd_layers):
                sgd_gram_norms[i] = _make_stats(_extract_curves(gdf, f"gram_norm_{i}_b"))

            for i in range(n_sgd_pairs):
                diff_curves = _extract_curves(gdf, f"balance_diff_{i}_b")
                sgd_balance_diffs_out[i] = _make_stats(diff_curves)
                gl = _extract_curves(gdf, f"gram_norm_{i}_b")
                gr = _extract_curves(gdf, f"gram_norm_{i + 1}_b")
                sgd_balance_ratios_out[i] = _make_stats_linear_ci(
                    diff_curves / np.maximum(gl + gr, 1e-30)
                )

            sgd_end_to_end_weight_norm = _make_stats(
                _extract_curves(gdf, "end_to_end_weight_norm_b")
            )

            # Cosine sim (per-run, derived from layer norms + param distance)
            if gd_param_norm is not None:
                sgd_pnorm_sq = sum(v ** 2 for v in sgd_ln_curves.values())
                gd_pnorm_sq = gd_param_norm ** 2
                dot = (gd_pnorm_sq[np.newaxis, :] + sgd_pnorm_sq - param_dist_curves ** 2) / 2.0
                sgd_pnorm = np.sqrt(sgd_pnorm_sq)
                denom = gd_param_norm[np.newaxis, :] * np.maximum(sgd_pnorm, 1e-30)
                cosine_sim = _make_stats_linear_ci(dot / denom)

        stats[key] = CompConfigStats(
            steps=steps,
            loss_gd=loss_gd,
            loss_sgd=loss_sgd,
            param_distance=param_distance,
            frobenius_distance=frobenius_distance,
            layer_distances=layer_distances,
            cosine_sim=cosine_sim,
            gd_param_norm=gd_param_norm,
            gd_end_to_end_weight_norm=gd_end_to_end_weight_norm,
            gd_layer_norms=gd_layer_norms,
            gd_gram_norms=gd_gram_norms,
            gd_balance_diffs=gd_balance_diffs,
            gd_balance_ratios=gd_balance_ratios,
            sgd_layer_norms=sgd_layer_norms,
            sgd_gram_norms=sgd_gram_norms,
            sgd_balance_diffs=sgd_balance_diffs_out,
            sgd_balance_ratios=sgd_balance_ratios_out,
            sgd_end_to_end_weight_norm=sgd_end_to_end_weight_norm,
        )

        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(
                f"\r  Stats: {idx + 1}/{total} ({100 * (idx + 1) / total:.0f}%)",
                end="", flush=True,
            )

    print(f"\n  {len(stats)} configurations computed")
    return stats


# =============================================================================
# Caching
# =============================================================================


def _ser_ms(ms: MetricStats | None) -> dict | None:
    return vars(ms) if ms is not None else None


def _de_ms(d: dict | None) -> MetricStats | None:
    return MetricStats(**d) if d is not None else None


def _ser_dict_ms(d: dict[int, MetricStats]) -> dict:
    return {k: vars(v) for k, v in d.items()}


def _de_dict_ms(d: dict) -> dict[int, MetricStats]:
    return {k: MetricStats(**v) for k, v in d.items()}


def _save_cache(stats: dict[tuple, CompConfigStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for key, cs in stats.items():
        data[key] = {
            "steps": cs.steps,
            "loss_gd": cs.loss_gd,
            "loss_sgd": _ser_ms(cs.loss_sgd),
            "param_distance": _ser_ms(cs.param_distance),
            "frobenius_distance": _ser_ms(cs.frobenius_distance),
            "layer_distances": _ser_dict_ms(cs.layer_distances),
            "cosine_sim": _ser_ms(cs.cosine_sim),
            "gd_param_norm": cs.gd_param_norm,
            "gd_end_to_end_weight_norm": cs.gd_end_to_end_weight_norm,
            "gd_layer_norms": dict(cs.gd_layer_norms),
            "gd_gram_norms": dict(cs.gd_gram_norms),
            "gd_balance_diffs": dict(cs.gd_balance_diffs),
            "gd_balance_ratios": dict(cs.gd_balance_ratios),
            "sgd_layer_norms": _ser_dict_ms(cs.sgd_layer_norms),
            "sgd_gram_norms": _ser_dict_ms(cs.sgd_gram_norms),
            "sgd_balance_diffs": _ser_dict_ms(cs.sgd_balance_diffs),
            "sgd_balance_ratios": _ser_dict_ms(cs.sgd_balance_ratios),
            "sgd_end_to_end_weight_norm": _ser_ms(cs.sgd_end_to_end_weight_norm),
        }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved to {path}")


def _load_cache(path: Path) -> dict[tuple, CompConfigStats] | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {
            key: CompConfigStats(
                steps=d["steps"],
                loss_gd=d["loss_gd"],
                loss_sgd=_de_ms(d["loss_sgd"]),
                param_distance=_de_ms(d["param_distance"]),
                frobenius_distance=_de_ms(d["frobenius_distance"]),
                layer_distances=_de_dict_ms(d["layer_distances"]),
                cosine_sim=_de_ms(d.get("cosine_sim")),
                gd_param_norm=d.get("gd_param_norm"),
                gd_end_to_end_weight_norm=d.get("gd_end_to_end_weight_norm"),
                gd_layer_norms=dict(d.get("gd_layer_norms", {})),
                gd_gram_norms=dict(d.get("gd_gram_norms", {})),
                gd_balance_diffs=dict(d.get("gd_balance_diffs", {})),
                gd_balance_ratios=dict(d.get("gd_balance_ratios", {})),
                sgd_layer_norms=_de_dict_ms(d.get("sgd_layer_norms", {})),
                sgd_gram_norms=_de_dict_ms(d.get("sgd_gram_norms", {})),
                sgd_balance_diffs=_de_dict_ms(d.get("sgd_balance_diffs", {})),
                sgd_balance_ratios=_de_dict_ms(d.get("sgd_balance_ratios", {})),
                sgd_end_to_end_weight_norm=_de_ms(d.get("sgd_end_to_end_weight_norm")),
            )
            for key, d in data.items()
        }
    except (pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"Warning: Failed to load cache ({e}), will recompute")
        return None


def get_stats(
    input_dir: Path,
    baseline_fallback: dict,
    cache_path: Path,
    force_recompute: bool = False,
) -> dict[tuple, CompConfigStats]:
    if not force_recompute:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"Loaded {len(cached)} configurations from cache")
            return cached

    stats = compute_all_stats(input_dir, baseline_fallback)
    _save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting Helpers
# =============================================================================


def _suptitle(
    title: str, gamma: float, noise: float,
    width: int, batch_size: int, n_seeds: int | str,
) -> str:
    gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    return (
        f"{title} | {gamma_name} (γ={gamma}) | noise={noise}"
        f" | width={width} | B={batch_size} | {fmt_seeds(n_seeds)} batch seeds"
    )


def _get_n_seeds(stats: dict[tuple, CompConfigStats]) -> int | str:
    key = next(iter(stats), None)
    return stats[key].loss_sgd.n if key else "?"


def _plot_loss_row(
    ax: plt.Axes, s: CompConfigStats, batch_size: int,
    mcfg: dict,
) -> None:
    """Plot baseline vs comparison loss on a single axes (used as top row of each figure)."""
    ax.plot(s.steps, s.loss_gd, label=mcfg["a_label"], color="C0", linewidth=1.5)
    ax.plot(
        s.steps, s.loss_sgd.mean,
        label=_b_label(mcfg, batch_size), color="C1", linewidth=1.5,
    )
    ax.fill_between(
        s.steps, s.loss_sgd.ci_lo, s.loss_sgd.ci_hi,
        alpha=0.3, color="C1",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Test loss")
    ax.legend(loc="upper right", fontsize=6)


def _plot_gd_vs_sgd_layers(
    ax: plt.Axes,
    steps: np.ndarray,
    gd_dict: dict[int, np.ndarray],
    sgd_dict: dict[int, MetricStats],
    ylabel: str,
    log_scale: bool = False,
    a_label: str = "GD",
    b_label: str = "SGD",
) -> None:
    """Plot per-layer baseline (solid) vs comparison (dashed + CI)."""
    for i in sorted(gd_dict.keys()):
        color = f"C{i}"
        ax.plot(
            steps, gd_dict[i], color=color, linewidth=1.5,
            label=f"L{i} {a_label}",
        )
        if i in sgd_dict:
            ms = sgd_dict[i]
            ax.plot(
                steps, ms.mean, color=color, linewidth=1.5,
                linestyle="--", label=f"L{i} {b_label}",
            )
            ax.fill_between(
                steps, ms.ci_lo, ms.ci_hi, alpha=0.15, color=color,
            )
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5, loc="best", ncols=2)


# =============================================================================
# Model Metrics Figure
# =============================================================================


def plot_model_metrics(
    stats: dict[tuple, CompConfigStats],
    gamma: float,
    noise: float,
    width: int,
    batch_size: int,
    model_seeds: list,
    mcfg: dict,
) -> plt.Figure:
    """6 rows: loss, layer norms, gram norms, balance diffs, balance ratio, eff weight norm."""
    n_cols = len(model_seeds)
    n_rows = 6
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    a_lbl = mcfg["a_label"]
    b_lbl = _b_short(mcfg, batch_size)

    for col, seed in enumerate(model_seeds):
        key = (width, gamma, noise, seed, batch_size)

        if key not in stats:
            axes[0, col].set_title(f"Model Seed {seed}")
            continue

        s = stats[key]

        # Row 0: Loss
        _plot_loss_row(axes[0, col], s, batch_size, mcfg)
        axes[0, col].set_title(f"Model Seed {seed}")

        # Row 1: Layer norms
        _plot_gd_vs_sgd_layers(
            axes[1, col], s.steps,
            s.gd_layer_norms, s.sgd_layer_norms,
            ylabel=r"$\|W_i\|_F$",
            a_label=a_lbl, b_label=b_lbl,
        )

        # Row 2: Gram norms
        _plot_gd_vs_sgd_layers(
            axes[2, col], s.steps,
            s.gd_gram_norms, s.sgd_gram_norms,
            ylabel=r"$\|W_i W_i^T\|_F$",
            a_label=a_lbl, b_label=b_lbl,
        )

        # Row 3: Balance diffs
        _plot_gd_vs_sgd_layers(
            axes[3, col], s.steps,
            s.gd_balance_diffs, s.sgd_balance_diffs,
            ylabel=r"$\|G_l\|_F$",
            a_label=a_lbl, b_label=b_lbl,
        )

        # Row 4: Balance ratio
        _plot_gd_vs_sgd_layers(
            axes[4, col], s.steps,
            s.gd_balance_ratios, s.sgd_balance_ratios,
            ylabel=r"$r_l$",
            a_label=a_lbl, b_label=b_lbl,
        )

        # Row 5: Effective weight norm
        ax = axes[5, col]
        if s.gd_end_to_end_weight_norm is not None:
            ax.plot(
                s.steps, s.gd_end_to_end_weight_norm,
                color="C0", linewidth=1.5, label=a_lbl,
            )
        if s.sgd_end_to_end_weight_norm is not None:
            ms = s.sgd_end_to_end_weight_norm
            ax.plot(
                s.steps, ms.mean,
                color="C1", linewidth=1.5, linestyle="--", label=b_lbl,
            )
            ax.fill_between(s.steps, ms.ci_lo, ms.ci_hi, alpha=0.2, color="C1")
        if col == 0:
            ax.set_ylabel(r"$\|W_{\mathrm{eff}}\|_F$")
        ax.set_xlabel("Training step")
        ax.legend(fontsize=6, loc="best")

    n_seeds = _get_n_seeds(stats)
    title_vs = f"{a_lbl} (solid) vs {b_lbl} (dashed)"
    fig.suptitle(
        _suptitle(f"Weight Structure: {title_vs}", gamma, noise, width, batch_size, n_seeds),
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Batch-Size Overlay Figure
# =============================================================================


BATCH_COLORS = {
    1: "C0", 2: "C1", 5: "C2", 10: "C3", 50: "C4",
}


def plot_distances(
    stats: dict[tuple, CompConfigStats],
    gamma: float,
    noise: float,
    width: int,
    batch_sizes: list[int],
    model_seeds: list,
    mcfg: dict,
) -> plt.Figure | None:
    """Distances figure: all batch sizes overlaid per model seed.

    Rows: loss, relative param distance, relative effective matrix distance,
    cosine similarity (optional), relative layer distances.
    """
    labels = _axis_labels(mcfg["sub_a"], mcfg["sub_b"])
    a_label = mcfg["a_label"]

    has_cosine = any(
        stats.get((width, gamma, noise, s, b), None) is not None
        and stats[(width, gamma, noise, s, b)].cosine_sim is not None
        for s in model_seeds for b in batch_sizes
    )

    # Detect number of layer distance columns
    n_layer_dists = 0
    for s in model_seeds:
        for b in batch_sizes:
            key = (width, gamma, noise, s, b)
            if key in stats:
                n_layer_dists = max(n_layer_dists, len(stats[key].layer_distances))

    n_cols = len(model_seeds)
    # loss + param + frob + [cosine] + layers
    n_rows = 3 + int(has_cosine) + int(n_layer_dists > 0)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    row_loss = 0
    row_param = 1
    row_frob = 2
    row_cosine = 3 if has_cosine else None
    row_layer = (3 + int(has_cosine)) if n_layer_dists > 0 else None

    cosine_data_min = np.inf
    cosine_data_max = -np.inf

    for col, seed in enumerate(model_seeds):
        # Loss row: baseline once + each batch size for comparison
        first_key = None
        for b in batch_sizes:
            k = (width, gamma, noise, seed, b)
            if k in stats:
                first_key = k
                break
        if first_key is not None:
            s0 = stats[first_key]
            axes[row_loss, col].plot(
                s0.steps, s0.loss_gd, label=a_label, color="black", linewidth=1.5,
            )

        for batch_size in batch_sizes:
            key = (width, gamma, noise, seed, batch_size)
            if key not in stats:
                continue
            s = stats[key]
            color = BATCH_COLORS.get(batch_size, "grey")
            lbl = f"B={batch_size}"

            # Loss
            ax = axes[row_loss, col]
            ax.plot(s.steps, s.loss_sgd.mean, color=color, linewidth=1.5, label=lbl)
            ax.fill_between(
                s.steps, s.loss_sgd.ci_lo, s.loss_sgd.ci_hi,
                alpha=0.15, color=color,
            )

            # Relative param distance
            ax = axes[row_param, col]
            if s.gd_param_norm is not None:
                norm = np.maximum(s.gd_param_norm, 1e-30)
                ax.plot(s.steps, s.param_distance.mean / norm,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps,
                    s.param_distance.ci_lo / norm,
                    s.param_distance.ci_hi / norm,
                    alpha=0.15, color=color,
                )
            else:
                ax.plot(s.steps, s.param_distance.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.param_distance.ci_lo, s.param_distance.ci_hi,
                    alpha=0.15, color=color,
                )

            # Relative effective matrix distance
            ax = axes[row_frob, col]
            if s.gd_end_to_end_weight_norm is not None:
                norm = np.maximum(s.gd_end_to_end_weight_norm, 1e-30)
                ax.plot(s.steps, s.frobenius_distance.mean / norm,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps,
                    s.frobenius_distance.ci_lo / norm,
                    s.frobenius_distance.ci_hi / norm,
                    alpha=0.15, color=color,
                )
            else:
                ax.plot(s.steps, s.frobenius_distance.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.frobenius_distance.ci_lo, s.frobenius_distance.ci_hi,
                    alpha=0.15, color=color,
                )

            # Cosine similarity
            if row_cosine is not None and s.cosine_sim is not None:
                ax = axes[row_cosine, col]
                ax.plot(s.steps, s.cosine_sim.mean,
                        color=color, linewidth=1.5, label=lbl)
                ax.fill_between(
                    s.steps, s.cosine_sim.ci_lo, s.cosine_sim.ci_hi,
                    alpha=0.15, color=color,
                )
                cosine_data_min = min(cosine_data_min, np.nanmin(s.cosine_sim.ci_lo))
                cosine_data_max = max(cosine_data_max, np.nanmax(s.cosine_sim.ci_hi))

            # Relative layer distances
            if row_layer is not None:
                ax = axes[row_layer, col]
                for i, ms in sorted(s.layer_distances.items()):
                    gd_ln = s.gd_layer_norms.get(i)
                    if gd_ln is not None:
                        norm = np.maximum(gd_ln, 1e-30)
                        ax.plot(s.steps, ms.mean / norm, color=color,
                                linewidth=1.0, linestyle=["-", "--", ":", "-."][i % 4],
                                label=f"{lbl} L{i}" if col == 0 else None)
                    else:
                        ax.plot(s.steps, ms.mean, color=color,
                                linewidth=1.0, linestyle=["-", "--", ":", "-."][i % 4],
                                label=f"{lbl} L{i}" if col == 0 else None)

        # Axis formatting (after all batch sizes plotted)
        axes[row_loss, col].set_title(f"Model Seed {seed}")
        axes[row_loss, col].set_yscale("log")
        axes[row_loss, col].set_ylabel("Test loss") if col == 0 else None
        axes[row_loss, col].legend(fontsize=5, loc="best", ncols=2)

        axes[row_param, col].set_yscale("log")
        axes[row_param, col].legend(fontsize=5, loc="best", ncols=2)
        if col == 0:
            any_key = next((k for k in ((width, gamma, noise, seed, b)
                           for b in batch_sizes) if k in stats), None)
            has_norm = any_key and stats[any_key].gd_param_norm is not None
            axes[row_param, col].set_ylabel(
                labels["param_rel"] if has_norm else labels["param_abs"]
            )

        axes[row_frob, col].set_yscale("log")
        axes[row_frob, col].legend(fontsize=5, loc="best", ncols=2)
        if col == 0:
            any_key = next((k for k in ((width, gamma, noise, seed, b)
                           for b in batch_sizes) if k in stats), None)
            has_eff = any_key and stats[any_key].gd_end_to_end_weight_norm is not None
            axes[row_frob, col].set_ylabel(
                labels["eff_rel"] if has_eff else labels["eff_abs"]
            )

        if row_cosine is not None:
            axes[row_cosine, col].axhline(
                1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8,
            )
            axes[row_cosine, col].legend(fontsize=5, loc="best", ncols=2)
            if col == 0:
                axes[row_cosine, col].set_ylabel(labels["cosine"])

        if row_layer is not None:
            axes[row_layer, col].set_yscale("log")
            axes[row_layer, col].set_xlabel("Training step")
            if col == 0:
                any_key = next((k for k in ((width, gamma, noise, seed, b)
                               for b in batch_sizes) if k in stats), None)
                has_ln = any_key and stats[any_key].gd_layer_norms
                axes[row_layer, col].set_ylabel(
                    labels["layer_rel"] if has_ln else labels["layer_abs"]
                )
                axes[row_layer, col].legend(fontsize=4, loc="best", ncols=4)

    # Synchronize cosine similarity y-axes across columns
    if row_cosine is not None and np.isfinite(cosine_data_min):
        margin = (cosine_data_max - cosine_data_min) * 0.05
        yl = cosine_data_min - margin
        yh = cosine_data_max + margin
        for c in range(n_cols):
            axes[row_cosine, c].set_ylim(yl, yh)
            axes[row_cosine, c].ticklabel_format(useOffset=False)

    gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    n_seeds = "?"
    for b in batch_sizes:
        for s in model_seeds:
            key = (width, gamma, noise, s, b)
            if key in stats:
                n_seeds = stats[key].loss_sgd.n
                break
        if n_seeds != "?":
            break

    fig.suptitle(
        f"Distances | {gamma_name} (γ={gamma}) | noise={noise}"
        f" | width={width} | {fmt_seeds(n_seeds)} batch seeds",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Batch-Seed Overlay Figure (individual runs)
# =============================================================================


@dataclass
class _SeedOverlayData:
    """Pre-extracted numpy arrays for one (width, gamma, noise, seed, batch_size) group."""
    steps: np.ndarray
    param_dist: np.ndarray       # (n_runs, n_steps)
    frob_dist: np.ndarray        # (n_runs, n_steps)
    cosine_sim: np.ndarray | None  # (n_runs, n_steps) or None
    gd_param_norm: np.ndarray | None
    gd_eff_norm: np.ndarray | None
    n_runs: int


def _precompute_seed_overlay_data(
    input_dir: Path,
    baseline_fallback: dict,
) -> dict[tuple, _SeedOverlayData]:
    """Load parquet once and pre-extract per-run curves for seed overlay plots."""
    comp_path = input_dir / "results.parquet"
    if not comp_path.exists():
        return {}

    print(f"  Loading comparative data for seed overlays...")
    df = pl.read_parquet(comp_path)

    n_layers = 0
    while f"layer_norm_{n_layers}_b" in df.columns:
        n_layers += 1

    groups = df.partition_by(GROUP_COLS, as_dict=True)
    result: dict[tuple, _SeedOverlayData] = {}

    for key, gdf in groups.items():
        if len(gdf) == 0:
            continue
        width, gamma, noise, seed, batch_size = key

        bl = baseline_fallback.get((width, gamma, noise, seed), {})
        gd_param_norm = bl.get("param_norm")
        gd_eff_norm = bl.get("end_to_end_weight_norm")
        if gd_param_norm is None:
            continue

        steps = np.array(gdf["step"][0])
        param_dist = _extract_curves(gdf, "param_distance")
        frob_dist = _extract_curves(gdf, "frobenius_distance")

        # Per-run cosine sim via polarization identity
        cosine_sim = None
        if n_layers > 0:
            sgd_pnorm_sq = sum(
                _extract_curves(gdf, f"layer_norm_{i}_b") ** 2
                for i in range(n_layers)
            )
            dot = (gd_param_norm ** 2 + sgd_pnorm_sq - param_dist ** 2) / 2.0
            sgd_pnorm = np.sqrt(sgd_pnorm_sq)
            denom = gd_param_norm * np.maximum(sgd_pnorm, 1e-30)
            cosine_sim = dot / denom

        result[key] = _SeedOverlayData(
            steps=steps,
            param_dist=param_dist,
            frob_dist=frob_dist,
            cosine_sim=cosine_sim,
            gd_param_norm=gd_param_norm,
            gd_eff_norm=gd_eff_norm,
            n_runs=len(gdf),
        )

    print(f"  {len(result)} groups pre-extracted for seed overlays")
    return result


def plot_seed_overlay(
    data: dict[tuple, _SeedOverlayData],
    gamma: float,
    noise: float,
    width: int,
    batch_size: int,
    model_seeds: list,
    mcfg: dict,
) -> plt.Figure | None:
    """Plot individual batch-seed runs overlaid.

    3 rows: relative param distance, cosine similarity, relative effective
    matrix distance.  One column per model seed.
    """
    labels = _axis_labels(mcfg["sub_a"], mcfg["sub_b"])

    has_cosine = any(
        data.get((width, gamma, noise, s, batch_size), None) is not None
        and data[(width, gamma, noise, s, batch_size)].cosine_sim is not None
        for s in model_seeds
    )
    has_frob = any(
        data.get((width, gamma, noise, s, batch_size), None) is not None
        and data[(width, gamma, noise, s, batch_size)].gd_eff_norm is not None
        for s in model_seeds
    )

    n_cols = len(model_seeds)
    n_rows = 1 + int(has_cosine) + int(has_frob)
    if n_rows == 0:
        return None

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False,
    )

    row_param = 0
    row_cosine = 1 if has_cosine else None
    row_frob = (1 + int(has_cosine)) if has_frob else None
    n_runs_any = 0
    cosine_data_min = np.inf
    cosine_data_max = -np.inf

    for col, seed in enumerate(model_seeds):
        key = (width, gamma, noise, seed, batch_size)
        d = data.get(key)
        if d is None:
            axes[row_param, col].set_title(f"Model Seed {seed}")
            continue

        n_runs_any = d.n_runs
        steps = d.steps

        # Relative param distance
        ax = axes[row_param, col]
        norm = np.maximum(d.gd_param_norm, 1e-30)
        rel = d.param_dist / norm
        for j in range(d.n_runs):
            ax.plot(steps, rel[j], color="C1", alpha=0.15, linewidth=0.5)
        ax.plot(steps, rel.mean(axis=0), color="black", linewidth=1.5, label="Mean")
        ax.set_yscale("log")
        ax.set_title(f"Model Seed {seed}")
        ax.legend(fontsize=6, loc="best")
        if col == 0:
            ax.set_ylabel(labels["param_rel"])

        # Cosine similarity
        if row_cosine is not None and d.cosine_sim is not None:
            ax = axes[row_cosine, col]
            for j in range(d.n_runs):
                ax.plot(steps, d.cosine_sim[j], color="C3", alpha=0.15, linewidth=0.5)
            ax.plot(steps, d.cosine_sim.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.legend(fontsize=6, loc="best")
            if col == 0:
                ax.set_ylabel(labels["cosine"])
            cosine_data_min = min(cosine_data_min, np.nanmin(d.cosine_sim))
            cosine_data_max = max(cosine_data_max, np.nanmax(d.cosine_sim))

        # Relative effective matrix distance
        if row_frob is not None and d.gd_eff_norm is not None:
            ax = axes[row_frob, col]
            norm = np.maximum(d.gd_eff_norm, 1e-30)
            rel = d.frob_dist / norm
            for j in range(d.n_runs):
                ax.plot(steps, rel[j], color="C2", alpha=0.15, linewidth=0.5)
            ax.plot(steps, rel.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            ax.set_yscale("log")
            ax.legend(fontsize=6, loc="best")
            ax.set_xlabel("Training step")
            if col == 0:
                ax.set_ylabel(labels["eff_rel"])

    # Synchronize cosine similarity y-axes across columns
    if row_cosine is not None and np.isfinite(cosine_data_min):
        margin = (cosine_data_max - cosine_data_min) * 0.05
        yl = cosine_data_min - margin
        yh = cosine_data_max + margin
        for c in range(n_cols):
            axes[row_cosine, c].set_ylim(yl, yh)
            axes[row_cosine, c].ticklabel_format(useOffset=False)

    gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    fig.suptitle(
        f"Individual Runs | {gamma_name} (γ={gamma}) | noise={noise}"
        f" | width={width} | B={batch_size} | {n_runs_any} runs",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


_seed_overlay_ctx: dict = {}


def _init_seed_overlay_worker(
    precomputed: dict, output_dir: str, mcfg: dict,
) -> None:
    _seed_overlay_ctx["data"] = precomputed
    _seed_overlay_ctx["output_dir"] = output_dir
    _seed_overlay_ctx["mcfg"] = mcfg


def _run_seed_overlay_task(task: tuple) -> None:
    gamma, noise, width, batch_size, model_seeds, filename = task
    data = _seed_overlay_ctx["data"]
    output_dir = Path(_seed_overlay_ctx["output_dir"])
    mcfg = _seed_overlay_ctx["mcfg"]

    fig = plot_seed_overlay(data, gamma, noise, width, batch_size, model_seeds, mcfg)
    if fig is None:
        return

    fig.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_seed_overlay_plots(
    input_dir: Path,
    baseline_fallback: dict,
    output_dir: Path,
    mcfg: dict,
) -> None:
    """Generate per-config seed overlay plots from raw parquet data."""
    precomputed = _precompute_seed_overlay_data(input_dir, baseline_fallback)
    if not precomputed:
        print("  Seed overlay: no data available, skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    widths = sorted({k[0] for k in precomputed})
    gammas = sorted({k[1] for k in precomputed})
    noise_levels = sorted({k[2] for k in precomputed})
    model_seeds = sorted({k[3] for k in precomputed})
    batch_sizes = sorted({k[4] for k in precomputed})

    tasks = []
    for gamma in gammas:
        for noise in noise_levels:
            for width in widths:
                for batch_size in batch_sizes:
                    tag = f"g{gamma}_noise{noise}_w{width}_b{batch_size}"
                    tasks.append((gamma, noise, width, batch_size,
                                  model_seeds, f"individual_runs_{tag}"))

    n_workers = min(os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} seed-overlay figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_seed_overlay_worker,
        initargs=(precomputed, str(output_dir), mcfg),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_seed_overlay_task, tasks), 1):
            if i % 10 == 0 or i == len(tasks):
                print(
                    f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                    end="", flush=True,
                )

    print(f"\nSeed-overlay plots saved to {output_dir}/")


# =============================================================================
# Parallel Plot Generation
# =============================================================================

_worker_ctx: dict = {}


def _init_plot_worker(all_stats: dict, output_dir: str, mcfg: dict) -> None:
    _worker_ctx["stats"] = all_stats
    _worker_ctx["output_dir"] = output_dir
    _worker_ctx["mcfg"] = mcfg


def _run_plot_task(task: tuple) -> None:
    plot_type, gamma, noise, width, batch_size, model_seeds, filename = task
    stats = _worker_ctx["stats"]
    output_dir = Path(_worker_ctx["output_dir"])
    mcfg = _worker_ctx["mcfg"]

    if plot_type == "distances":
        # batch_size field carries the full batch_sizes list
        fig = plot_distances(stats, gamma, noise, width, batch_size, model_seeds, mcfg)
    elif plot_type == "model_metrics":
        fig = plot_model_metrics(stats, gamma, noise, width, batch_size, model_seeds, mcfg)
    else:
        return

    if fig is None:
        return

    fig.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    stats: dict[tuple, CompConfigStats],
    output_dir: Path,
    mcfg: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive parameter values from stats keys:
    # (width, gamma, noise, model_seed, batch_size)
    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    # Check if any config has model B metrics (for model_metrics figure)
    has_sgd_metrics = any(len(cs.sgd_layer_norms) > 0 for cs in stats.values())

    tasks = []
    for gamma in gammas:
        for noise in noise_levels:
            for width in widths:
                # Distances figure (one per gamma/noise/width, all batch sizes overlaid)
                tag = f"g{gamma}_noise{noise}_w{width}"
                tasks.append(("distances", gamma, noise, width,
                              batch_sizes, model_seeds,
                              f"distances_{tag}"))

                # Model metrics (one per batch size)
                if has_sgd_metrics:
                    for batch_size in batch_sizes:
                        tag_b = f"g{gamma}_noise{noise}_w{width}_b{batch_size}"
                        common = (gamma, noise, width, batch_size, model_seeds)
                        tasks.append(("model_metrics", *common, f"model_metrics_{tag_b}"))

    if not has_sgd_metrics:
        print("  Note: model B metrics not available — skipping model_metrics figures.")
        print("  Re-run the comparative sweep with model metrics enabled.")

    n_workers = min(os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_plot_worker,
        initargs=(stats, str(output_dir), mcfg),
    ) as pool:
        for i, _ in enumerate(pool.imap_unordered(_run_plot_task, tasks), 1):
            print(
                f"\r  Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%)",
                end="", flush=True,
            )

    print(f"\nAll plots saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparative metrics and model metrics (offline or online)",
    )
    parser.add_argument(
        "--mode", choices=["offline", "online"], default="offline",
        help="Experiment mode: offline (GD vs SGD) or online (large-batch vs mini-batch)",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Comparative results directory",
    )
    parser.add_argument(
        "--baseline-input", type=Path, default=None,
        help="Baseline model metrics directory (for normalisation)",
    )
    # Keep --gd-input as hidden alias for backward compatibility
    parser.add_argument("--gd-input", type=Path, default=None, dest="gd_input_compat",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output figures directory",
    )
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Cache file for computed statistics",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    args = parser.parse_args()

    mcfg = MODE_CONFIGS[args.mode]
    input_dir = args.input or mcfg["input"]
    baseline_dir = args.baseline_input or args.gd_input_compat or mcfg["baseline_input"]
    output_dir = args.output or mcfg["output"]
    cache_path = args.cache or mcfg["cache"]

    print(f"Mode:      {args.mode}")
    print(f"Input:     {input_dir}")
    print(f"Baseline:  {baseline_dir}")
    print(f"Output:    {output_dir}")

    baseline_fallback = _load_baseline_fallback(baseline_dir)

    stats = get_stats(
        input_dir, baseline_fallback,
        cache_path, args.recompute,
    )
    generate_all_plots(stats, output_dir, mcfg)
    generate_seed_overlay_plots(input_dir, baseline_fallback, output_dir, mcfg)

    print("\nDone!")


if __name__ == "__main__":
    main()
