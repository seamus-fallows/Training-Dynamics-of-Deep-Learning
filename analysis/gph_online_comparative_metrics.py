"""
GPH Online Comparative Metrics Analysis

Plots large-batch (B=500) vs mini-batch comparative metrics and model metrics
in the online (infinite data) regime, with mini-batch quantities averaged over
batch seeds.

Two figure types per (gamma, noise, width, batch_size) configuration:

  - Distances: loss context, relative param/frobenius/layer distances, cosine similarity
  - Model Metrics: loss context, layer norms, gram norms, balance diffs/ratio,
    effective weight norm  (B=500 solid vs mini-batch dashed + 95% CI)

Data sources:
  - Comparative sweep (--input): test_loss_a/b, param_distance, layer_distances,
    frobenius_distance, plus both model_a and model_b metrics (both stochastic).
  - Baseline sweep (--baseline-input): layer_norms, gram_norms, balance_diffs,
    end_to_end_weight_norm for the B=500 model (averaged over batch seeds).

Usage (run from the project root):

    python analysis/gph_online_comparative_metrics.py --recompute

Options:
    --input PATH           Override comparative input dir
    --baseline-input PATH  Override baseline metrics input dir
    --output PATH          Override output figures dir
    --cache PATH           Override cache file path
    --recompute            Force recompute statistics (ignore cache)
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

from _common import CACHE_DIR, GAMMA_NAMES, BL_KEY_COLS, fmt_seeds


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT = Path("outputs/gph_online_comparative_metrics/comparative")
DEFAULT_BASELINE_INPUT = Path("outputs/gph_online_comparative_metrics/baseline_metrics")
DEFAULT_OUTPUT = Path("figures/gph_online_comparative_metrics")
DEFAULT_CACHE = CACHE_DIR / "gph_online_comparative_metrics.pkl"

# Columns that define a unique configuration (batch_seed is averaged over)
GROUP_COLS = [
    "model.hidden_dim", "model.gamma", "data.noise_std",
    "model.model_seed", "training_b.batch_size",
]


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

    In online mode, the baseline (B=500) is also stochastic, so baseline
    metrics are stored as MetricStats with confidence intervals.
    """

    steps: np.ndarray

    # Loss
    loss_baseline: MetricStats
    loss_sgd: MetricStats

    # Comparative distances
    param_distance: MetricStats
    frobenius_distance: MetricStats
    layer_distances: dict[int, MetricStats]

    # Cosine similarity
    cosine_sim: MetricStats | None

    # Baseline model metrics (from separate sweep, averaged over batch seeds)
    bl_param_norm: np.ndarray | None
    bl_end_to_end_weight_norm: MetricStats | None
    bl_layer_norms: dict[int, MetricStats]
    bl_gram_norms: dict[int, MetricStats]
    bl_balance_diffs: dict[int, MetricStats]
    bl_balance_ratios: dict[int, MetricStats]

    # SGD model metrics (from _b columns in comparative data)
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
    """Build MetricStats with linear (not log-space) 95% CI."""
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
# Baseline Metrics Loading
# =============================================================================


def _load_baseline_metrics(
    baseline_dir: Path,
) -> dict[tuple, dict]:
    """Load baseline (B=500) model metrics averaged over batch seeds.

    Returns a dict keyed by (width, gamma, noise, seed) with layer norms,
    gram norms, balance diffs/ratios, param norm, and effective weight norm,
    each as MetricStats (since the baseline is stochastic in online mode).
    """
    path = baseline_dir / "results.parquet"
    if not path.exists():
        print(f"  Baseline metrics not found at {path}")
        return {}

    print(f"  Loading baseline metrics from {path}...")
    df = pl.read_parquet(path)

    # Detect layers
    n_layers = 0
    while f"layer_norm_{n_layers}" in df.columns:
        n_layers += 1
    n_pairs = max(n_layers - 1, 0)

    groups = df.partition_by(BL_KEY_COLS, as_dict=True)
    result = {}

    for key, gdf in groups.items():
        if len(gdf) == 0:
            continue

        layer_norms = {}
        gram_norms = {}
        for i in range(n_layers):
            col = f"layer_norm_{i}"
            if col in gdf.columns:
                layer_norms[i] = _make_stats(_extract_curves(gdf, col))
            col = f"gram_norm_{i}"
            if col in gdf.columns:
                gram_norms[i] = _make_stats(_extract_curves(gdf, col))

        balance_diffs = {}
        balance_ratios = {}
        for i in range(n_pairs):
            col = f"balance_diff_{i}"
            if col in gdf.columns:
                diff_curves = _extract_curves(gdf, col)
                balance_diffs[i] = _make_stats(diff_curves)
                gcol_l = f"gram_norm_{i}"
                gcol_r = f"gram_norm_{i + 1}"
                if gcol_l in gdf.columns and gcol_r in gdf.columns:
                    gl = _extract_curves(gdf, gcol_l)
                    gr = _extract_curves(gdf, gcol_r)
                    balance_ratios[i] = _make_stats_linear_ci(
                        diff_curves / np.maximum(gl + gr, 1e-30)
                    )

        # Mean param norm (for normalisation of distances)
        param_norm = None
        if layer_norms:
            # Use the mean of each layer norm to compute a single param norm curve
            param_norm = np.sqrt(sum(v.mean ** 2 for v in layer_norms.values()))

        eff = None
        if "end_to_end_weight_norm" in gdf.columns:
            eff = _make_stats(_extract_curves(gdf, "end_to_end_weight_norm"))

        result[key] = {
            "layer_norms": layer_norms,
            "gram_norms": gram_norms,
            "balance_diffs": balance_diffs,
            "balance_ratios": balance_ratios,
            "param_norm": param_norm,
            "end_to_end_weight_norm": eff,
        }

    print(f"  {len(result)} baseline configs loaded")
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
    i = 0
    while f"layer_norm_{i}_b" in df.columns:
        i += 1
    return i


def compute_all_stats(
    input_dir: Path,
    bl_fallback: dict,
) -> dict[tuple, CompConfigStats]:
    """Load comparative results and compute per-config stats over batch seeds."""
    path = input_dir / "results.parquet"
    print(f"Loading {path}...")
    df = pl.read_parquet(path)
    print(f"  {len(df)} runs loaded")

    n_comp_layers = _detect_comp_layers(df)
    n_sgd_layers = _detect_sgd_model_layers(df)
    has_sgd_metrics = n_sgd_layers > 0
    n_sgd_pairs = max(n_sgd_layers - 1, 0)

    # Check for _a model metrics too
    n_a_layers = 0
    while f"layer_norm_{n_a_layers}_a" in df.columns:
        n_a_layers += 1

    print(f"  {n_comp_layers} layer distance cols")
    if has_sgd_metrics:
        print(f"  {n_sgd_layers} SGD model layers (_b columns)")
    if n_a_layers > 0:
        print(f"  {n_a_layers} baseline model layers (_a columns)")

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
        loss_baseline = _make_stats(_extract_curves(gdf, "test_loss_a"))
        loss_sgd = _make_stats(_extract_curves(gdf, "test_loss_b"))

        # -- Comparative distances --
        param_dist_curves = _extract_curves(gdf, "param_distance")
        param_distance = _make_stats(param_dist_curves)
        frobenius_distance = _make_stats(_extract_curves(gdf, "frobenius_distance"))

        layer_distances: dict[int, MetricStats] = {}
        for i in range(n_comp_layers):
            layer_distances[i] = _make_stats(_extract_curves(gdf, f"layer_distance_{i}"))

        # -- Baseline model metrics (from fallback) --
        bl_key = (width, gamma, noise, seed)
        bl = bl_fallback.get(bl_key, {})
        bl_layer_norms = bl.get("layer_norms", {})
        bl_gram_norms = bl.get("gram_norms", {})
        bl_balance_diffs = bl.get("balance_diffs", {})
        bl_balance_ratios = bl.get("balance_ratios", {})
        bl_param_norm = bl.get("param_norm")
        bl_end_to_end_weight_norm = bl.get("end_to_end_weight_norm")

        # -- SGD model metrics & cosine sim --
        sgd_layer_norms: dict[int, MetricStats] = {}
        sgd_gram_norms: dict[int, MetricStats] = {}
        sgd_balance_diffs_out: dict[int, MetricStats] = {}
        sgd_balance_ratios_out: dict[int, MetricStats] = {}
        sgd_end_to_end_weight_norm: MetricStats | None = None
        cosine_sim: MetricStats | None = None

        if has_sgd_metrics:
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
            # Use exact per-run model_a norms from _a columns when available
            if n_a_layers > 0:
                a_pnorm_sq = sum(
                    _extract_curves(gdf, f"layer_norm_{i}_a") ** 2
                    for i in range(n_a_layers)
                )
                sgd_pnorm_sq = sum(v ** 2 for v in sgd_ln_curves.values())
                dot = (a_pnorm_sq + sgd_pnorm_sq - param_dist_curves ** 2) / 2.0
                a_pnorm = np.sqrt(a_pnorm_sq)
                sgd_pnorm = np.sqrt(sgd_pnorm_sq)
                denom = a_pnorm * np.maximum(sgd_pnorm, 1e-30)
                cosine_sim = _make_stats_linear_ci(dot / denom)
            elif bl_param_norm is not None:
                sgd_pnorm_sq = sum(v ** 2 for v in sgd_ln_curves.values())
                bl_pnorm_sq = bl_param_norm ** 2
                dot = (bl_pnorm_sq[np.newaxis, :] + sgd_pnorm_sq - param_dist_curves ** 2) / 2.0
                sgd_pnorm = np.sqrt(sgd_pnorm_sq)
                denom = bl_param_norm[np.newaxis, :] * np.maximum(sgd_pnorm, 1e-30)
                cosine_sim = _make_stats_linear_ci(dot / denom)

        stats[key] = CompConfigStats(
            steps=steps,
            loss_baseline=loss_baseline,
            loss_sgd=loss_sgd,
            param_distance=param_distance,
            frobenius_distance=frobenius_distance,
            layer_distances=layer_distances,
            cosine_sim=cosine_sim,
            bl_param_norm=bl_param_norm,
            bl_end_to_end_weight_norm=bl_end_to_end_weight_norm,
            bl_layer_norms=bl_layer_norms,
            bl_gram_norms=bl_gram_norms,
            bl_balance_diffs=bl_balance_diffs,
            bl_balance_ratios=bl_balance_ratios,
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
            "loss_baseline": _ser_ms(cs.loss_baseline),
            "loss_sgd": _ser_ms(cs.loss_sgd),
            "param_distance": _ser_ms(cs.param_distance),
            "frobenius_distance": _ser_ms(cs.frobenius_distance),
            "layer_distances": _ser_dict_ms(cs.layer_distances),
            "cosine_sim": _ser_ms(cs.cosine_sim),
            "bl_param_norm": cs.bl_param_norm,
            "bl_end_to_end_weight_norm": _ser_ms(cs.bl_end_to_end_weight_norm),
            "bl_layer_norms": _ser_dict_ms(cs.bl_layer_norms),
            "bl_gram_norms": _ser_dict_ms(cs.bl_gram_norms),
            "bl_balance_diffs": _ser_dict_ms(cs.bl_balance_diffs),
            "bl_balance_ratios": _ser_dict_ms(cs.bl_balance_ratios),
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
                loss_baseline=_de_ms(d["loss_baseline"]),
                loss_sgd=_de_ms(d["loss_sgd"]),
                param_distance=_de_ms(d["param_distance"]),
                frobenius_distance=_de_ms(d["frobenius_distance"]),
                layer_distances=_de_dict_ms(d["layer_distances"]),
                cosine_sim=_de_ms(d.get("cosine_sim")),
                bl_param_norm=d.get("bl_param_norm"),
                bl_end_to_end_weight_norm=_de_ms(d.get("bl_end_to_end_weight_norm")),
                bl_layer_norms=_de_dict_ms(d.get("bl_layer_norms", {})),
                bl_gram_norms=_de_dict_ms(d.get("bl_gram_norms", {})),
                bl_balance_diffs=_de_dict_ms(d.get("bl_balance_diffs", {})),
                bl_balance_ratios=_de_dict_ms(d.get("bl_balance_ratios", {})),
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
    bl_fallback: dict,
    cache_path: Path,
    force_recompute: bool = False,
) -> dict[tuple, CompConfigStats]:
    if not force_recompute:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"Loaded {len(cached)} configurations from cache")
            return cached

    stats = compute_all_stats(input_dir, bl_fallback)
    _save_cache(stats, cache_path)
    return stats


# =============================================================================
# Plotting Helpers
# =============================================================================

BL_LABEL = "B=500"


def _suptitle(
    title: str, gamma: float, noise: float,
    width: int, batch_size: int, n_seeds: int | str,
) -> str:
    gamma_name = GAMMA_NAMES.get(gamma, f"γ={gamma}")
    return (
        f"{title} | {gamma_name} (γ={gamma}) | noise={noise}"
        f" | width={width} | B={batch_size} | {fmt_seeds(n_seeds)} batch seeds"
        f" | Online (infinite data)"
    )


def _get_n_seeds(stats: dict[tuple, CompConfigStats]) -> int | str:
    key = next(iter(stats), None)
    return stats[key].loss_sgd.n if key else "?"


def _plot_loss_row(ax: plt.Axes, s: CompConfigStats, batch_size: int) -> None:
    """Plot baseline vs SGD loss on a single axes."""
    ax.plot(
        s.steps, s.loss_baseline.mean,
        label=BL_LABEL, color="C0", linewidth=1.5,
    )
    ax.fill_between(
        s.steps, s.loss_baseline.ci_lo, s.loss_baseline.ci_hi,
        alpha=0.3, color="C0",
    )
    ax.plot(
        s.steps, s.loss_sgd.mean,
        label=f"B={batch_size}", color="C1", linewidth=1.5,
    )
    ax.fill_between(
        s.steps, s.loss_sgd.ci_lo, s.loss_sgd.ci_hi,
        alpha=0.3, color="C1",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Test loss")
    ax.legend(loc="upper right", fontsize=6)


def _plot_bl_vs_sgd_layers(
    ax: plt.Axes,
    steps: np.ndarray,
    bl_dict: dict[int, MetricStats],
    sgd_dict: dict[int, MetricStats],
    ylabel: str,
    sgd_label: str = "SGD",
    log_scale: bool = False,
) -> None:
    """Plot per-layer baseline (solid + CI) vs SGD (dashed + CI) comparison."""
    for i in sorted(bl_dict.keys()):
        color = f"C{i}"
        bl_ms = bl_dict[i]
        ax.plot(
            steps, bl_ms.mean, color=color, linewidth=1.5,
            label=f"L{i} {BL_LABEL}",
        )
        ax.fill_between(
            steps, bl_ms.ci_lo, bl_ms.ci_hi, alpha=0.1, color=color,
        )
        if i in sgd_dict:
            ms = sgd_dict[i]
            ax.plot(
                steps, ms.mean, color=color, linewidth=1.5,
                linestyle="--", label=f"L{i} {sgd_label}",
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
) -> plt.Figure:
    """6 rows: loss, layer norms, gram norms, balance diffs, balance ratio, eff weight norm."""
    n_cols = len(model_seeds)
    n_rows = 6
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), squeeze=False,
    )

    for col, seed in enumerate(model_seeds):
        key = (width, gamma, noise, seed, batch_size)

        if key not in stats:
            axes[0, col].set_title(f"Model Seed {seed}")
            continue

        s = stats[key]
        sgd_lbl = f"B={batch_size}"

        # Row 0: Loss
        _plot_loss_row(axes[0, col], s, batch_size)
        axes[0, col].set_title(f"Model Seed {seed}")

        # Row 1: Layer norms
        _plot_bl_vs_sgd_layers(
            axes[1, col], s.steps,
            s.bl_layer_norms, s.sgd_layer_norms,
            ylabel=r"$\|W_i\|_F$", sgd_label=sgd_lbl,
        )

        # Row 2: Gram norms
        _plot_bl_vs_sgd_layers(
            axes[2, col], s.steps,
            s.bl_gram_norms, s.sgd_gram_norms,
            ylabel=r"$\|W_i W_i^T\|_F$", sgd_label=sgd_lbl,
        )

        # Row 3: Balance diffs
        _plot_bl_vs_sgd_layers(
            axes[3, col], s.steps,
            s.bl_balance_diffs, s.sgd_balance_diffs,
            ylabel=r"$\|G_l\|_F$", sgd_label=sgd_lbl,
        )

        # Row 4: Balance ratio
        _plot_bl_vs_sgd_layers(
            axes[4, col], s.steps,
            s.bl_balance_ratios, s.sgd_balance_ratios,
            ylabel=r"$r_l$", sgd_label=sgd_lbl,
        )

        # Row 5: Effective weight norm
        ax = axes[5, col]
        if s.bl_end_to_end_weight_norm is not None:
            bl_ms = s.bl_end_to_end_weight_norm
            ax.plot(
                s.steps, bl_ms.mean,
                color="C0", linewidth=1.5, label=BL_LABEL,
            )
            ax.fill_between(
                s.steps, bl_ms.ci_lo, bl_ms.ci_hi, alpha=0.2, color="C0",
            )
        if s.sgd_end_to_end_weight_norm is not None:
            ms = s.sgd_end_to_end_weight_norm
            ax.plot(
                s.steps, ms.mean,
                color="C1", linewidth=1.5, linestyle="--", label=sgd_lbl,
            )
            ax.fill_between(s.steps, ms.ci_lo, ms.ci_hi, alpha=0.2, color="C1")
        if col == 0:
            ax.set_ylabel(r"$\|W_{\mathrm{eff}}\|_F$")
        ax.set_xlabel("Training step")
        ax.legend(fontsize=6, loc="best")

    n_seeds = _get_n_seeds(stats)
    fig.suptitle(
        _suptitle(
            f"Weight Structure: {BL_LABEL} (solid) vs B={batch_size} (dashed)",
            gamma, noise, width, batch_size, n_seeds,
        ),
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
) -> plt.Figure | None:
    """Distances figure: all batch sizes overlaid per model seed.

    Rows: loss, relative param distance, relative effective matrix distance,
    cosine similarity (optional), relative layer distances.
    """
    has_cosine = any(
        stats.get((width, gamma, noise, s, b), None) is not None
        and stats[(width, gamma, noise, s, b)].cosine_sim is not None
        for s in model_seeds for b in batch_sizes
    )

    n_layer_dists = 0
    for s in model_seeds:
        for b in batch_sizes:
            key = (width, gamma, noise, s, b)
            if key in stats:
                n_layer_dists = max(n_layer_dists, len(stats[key].layer_distances))

    n_cols = len(model_seeds)
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
        # Loss row: baseline once + each batch size
        first_key = None
        for b in batch_sizes:
            k = (width, gamma, noise, seed, b)
            if k in stats:
                first_key = k
                break
        if first_key is not None:
            s0 = stats[first_key]
            axes[row_loss, col].plot(
                s0.steps, s0.loss_baseline.mean,
                label=BL_LABEL, color="black", linewidth=1.5,
            )
            axes[row_loss, col].fill_between(
                s0.steps, s0.loss_baseline.ci_lo, s0.loss_baseline.ci_hi,
                alpha=0.15, color="black",
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
            if s.bl_param_norm is not None:
                norm = np.maximum(s.bl_param_norm, 1e-30)
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
            if s.bl_end_to_end_weight_norm is not None:
                norm = np.maximum(s.bl_end_to_end_weight_norm.mean, 1e-30)
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
                    bl_ln = s.bl_layer_norms.get(i)
                    if bl_ln is not None:
                        norm = np.maximum(bl_ln.mean, 1e-30)
                        ax.plot(s.steps, ms.mean / norm, color=color,
                                linewidth=1.0, linestyle=["-", "--", ":", "-."][i % 4],
                                label=f"{lbl} L{i}" if col == 0 else None)
                    else:
                        ax.plot(s.steps, ms.mean, color=color,
                                linewidth=1.0, linestyle=["-", "--", ":", "-."][i % 4],
                                label=f"{lbl} L{i}" if col == 0 else None)

        # Axis formatting
        axes[row_loss, col].set_title(f"Model Seed {seed}")
        axes[row_loss, col].set_yscale("log")
        axes[row_loss, col].set_ylabel("Test loss") if col == 0 else None
        axes[row_loss, col].legend(fontsize=5, loc="best", ncols=2)

        axes[row_param, col].set_yscale("log")
        axes[row_param, col].legend(fontsize=5, loc="best", ncols=2)
        if col == 0:
            any_key = next((k for k in ((width, gamma, noise, seed, b)
                           for b in batch_sizes) if k in stats), None)
            has_norm = any_key and stats[any_key].bl_param_norm is not None
            if has_norm:
                axes[row_param, col].set_ylabel(
                    r"$\|\theta_{\mathrm{SGD}} - \theta_{\mathrm{BL}}\|"
                    r" / \|\theta_{\mathrm{BL}}\|$"
                )
            else:
                axes[row_param, col].set_ylabel(
                    r"$\|\theta_{\mathrm{SGD}} - \theta_{\mathrm{BL}}\|$"
                )

        axes[row_frob, col].set_yscale("log")
        axes[row_frob, col].legend(fontsize=5, loc="best", ncols=2)
        if col == 0:
            any_key = next((k for k in ((width, gamma, noise, seed, b)
                           for b in batch_sizes) if k in stats), None)
            has_eff = any_key and stats[any_key].bl_end_to_end_weight_norm is not None
            if has_eff:
                axes[row_frob, col].set_ylabel(
                    r"$\|W_{\mathrm{eff}}^{\mathrm{SGD}} - W_{\mathrm{eff}}^{\mathrm{BL}}\|_F"
                    r" / \|W_{\mathrm{eff}}^{\mathrm{BL}}\|_F$"
                )
            else:
                axes[row_frob, col].set_ylabel(
                    r"$\|W_{\mathrm{eff}}^{\mathrm{SGD}} - W_{\mathrm{eff}}^{\mathrm{BL}}\|_F$"
                )

        if row_cosine is not None:
            axes[row_cosine, col].axhline(
                1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8,
            )
            axes[row_cosine, col].legend(fontsize=5, loc="best", ncols=2)
            if col == 0:
                axes[row_cosine, col].set_ylabel(
                    r"$\cos(\theta_{\mathrm{BL}}, \theta_{\mathrm{SGD}})$"
                )

        if row_layer is not None:
            axes[row_layer, col].set_yscale("log")
            axes[row_layer, col].set_xlabel("Training step")
            if col == 0:
                any_key = next((k for k in ((width, gamma, noise, seed, b)
                               for b in batch_sizes) if k in stats), None)
                has_ln = any_key and stats[any_key].bl_layer_norms
                if has_ln:
                    axes[row_layer, col].set_ylabel(
                        r"$\|W_i^{\mathrm{SGD}} - W_i^{\mathrm{BL}}\|_F"
                        r" / \|W_i^{\mathrm{BL}}\|_F$"
                    )
                else:
                    axes[row_layer, col].set_ylabel(
                        r"$\|W_i^{\mathrm{SGD}} - W_i^{\mathrm{BL}}\|_F$"
                    )
                axes[row_layer, col].legend(fontsize=4, loc="best", ncols=4)

    # Synchronize cosine similarity y-axes
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
        f" | width={width} | {fmt_seeds(n_seeds)} batch seeds"
        f" | Online (infinite data)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


# =============================================================================
# Seed Overlay Figure
# =============================================================================


@dataclass
class _SeedOverlayData:
    steps: np.ndarray
    param_dist: np.ndarray       # (n_runs, n_steps)
    frob_dist: np.ndarray        # (n_runs, n_steps)
    cosine_sim: np.ndarray | None  # (n_runs, n_steps) or None
    bl_param_norm: np.ndarray | None   # (n_steps,) mean baseline param norm
    bl_eff_norm: np.ndarray | None     # (n_steps,) mean baseline effective norm
    n_runs: int


def _precompute_seed_overlay_data(
    input_dir: Path,
    bl_fallback: dict,
) -> dict[tuple, _SeedOverlayData]:
    comp_path = input_dir / "results.parquet"
    if not comp_path.exists():
        return {}

    print(f"  Loading comparative data for seed overlays...")
    df = pl.read_parquet(comp_path)

    n_b_layers = 0
    while f"layer_norm_{n_b_layers}_b" in df.columns:
        n_b_layers += 1
    n_a_layers = 0
    while f"layer_norm_{n_a_layers}_a" in df.columns:
        n_a_layers += 1

    groups = df.partition_by(GROUP_COLS, as_dict=True)
    result: dict[tuple, _SeedOverlayData] = {}

    for key, gdf in groups.items():
        if len(gdf) == 0:
            continue
        width, gamma, noise, seed, batch_size = key

        bl = bl_fallback.get((width, gamma, noise, seed), {})
        bl_param_norm = bl.get("param_norm")
        bl_eff_norm_ms = bl.get("end_to_end_weight_norm")
        bl_eff_norm = bl_eff_norm_ms.mean if bl_eff_norm_ms is not None else None

        steps = np.array(gdf["step"][0])
        param_dist = _extract_curves(gdf, "param_distance")
        frob_dist = _extract_curves(gdf, "frobenius_distance")

        # Compute per-run model_a param norm from _a columns (exact)
        a_param_norm = None
        if n_a_layers > 0:
            a_pnorm_sq = sum(
                _extract_curves(gdf, f"layer_norm_{i}_a") ** 2
                for i in range(n_a_layers)
            )
            a_param_norm = np.sqrt(a_pnorm_sq)
        elif bl_param_norm is not None:
            a_param_norm = bl_param_norm[np.newaxis, :].repeat(len(gdf), axis=0)

        if a_param_norm is None:
            continue

        # Use mean over runs for normalisation of distances
        norm_param = a_param_norm.mean(axis=0) if a_param_norm.ndim == 2 else a_param_norm

        cosine_sim = None
        if n_b_layers > 0 and a_param_norm is not None:
            sgd_pnorm_sq = sum(
                _extract_curves(gdf, f"layer_norm_{i}_b") ** 2
                for i in range(n_b_layers)
            )
            if n_a_layers > 0:
                a_pnorm_sq = sum(
                    _extract_curves(gdf, f"layer_norm_{i}_a") ** 2
                    for i in range(n_a_layers)
                )
            else:
                a_pnorm_sq = bl_param_norm[np.newaxis, :] ** 2
            dot = (a_pnorm_sq + sgd_pnorm_sq - param_dist ** 2) / 2.0
            a_pnorm = np.sqrt(a_pnorm_sq)
            sgd_pnorm = np.sqrt(sgd_pnorm_sq)
            denom = a_pnorm * np.maximum(sgd_pnorm, 1e-30)
            cosine_sim = dot / denom

        result[key] = _SeedOverlayData(
            steps=steps,
            param_dist=param_dist,
            frob_dist=frob_dist,
            cosine_sim=cosine_sim,
            bl_param_norm=norm_param,
            bl_eff_norm=bl_eff_norm,
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
) -> plt.Figure | None:
    has_cosine = any(
        data.get((width, gamma, noise, s, batch_size), None) is not None
        and data[(width, gamma, noise, s, batch_size)].cosine_sim is not None
        for s in model_seeds
    )
    has_frob = any(
        data.get((width, gamma, noise, s, batch_size), None) is not None
        and data[(width, gamma, noise, s, batch_size)].bl_eff_norm is not None
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
        norm = np.maximum(d.bl_param_norm, 1e-30)
        rel = d.param_dist / norm
        for j in range(d.n_runs):
            ax.plot(steps, rel[j], color="C1", alpha=0.15, linewidth=0.5)
        ax.plot(steps, rel.mean(axis=0), color="black", linewidth=1.5, label="Mean")
        ax.set_yscale("log")
        ax.set_title(f"Model Seed {seed}")
        ax.legend(fontsize=6, loc="best")
        if col == 0:
            ax.set_ylabel(
                r"$\|\theta_{\mathrm{SGD}} - \theta_{\mathrm{BL}}\|"
                r" / \|\theta_{\mathrm{BL}}\|$"
            )

        # Cosine similarity
        if row_cosine is not None and d.cosine_sim is not None:
            ax = axes[row_cosine, col]
            for j in range(d.n_runs):
                ax.plot(steps, d.cosine_sim[j], color="C3", alpha=0.15, linewidth=0.5)
            ax.plot(steps, d.cosine_sim.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.legend(fontsize=6, loc="best")
            if col == 0:
                ax.set_ylabel(r"$\cos(\theta_{\mathrm{BL}}, \theta_{\mathrm{SGD}})$")
            cosine_data_min = min(cosine_data_min, np.nanmin(d.cosine_sim))
            cosine_data_max = max(cosine_data_max, np.nanmax(d.cosine_sim))

        # Relative effective matrix distance
        if row_frob is not None and d.bl_eff_norm is not None:
            ax = axes[row_frob, col]
            norm = np.maximum(d.bl_eff_norm, 1e-30)
            rel = d.frob_dist / norm
            for j in range(d.n_runs):
                ax.plot(steps, rel[j], color="C2", alpha=0.15, linewidth=0.5)
            ax.plot(steps, rel.mean(axis=0), color="black", linewidth=1.5, label="Mean")
            ax.set_yscale("log")
            ax.legend(fontsize=6, loc="best")
            ax.set_xlabel("Training step")
            if col == 0:
                ax.set_ylabel(
                    r"$\|W_{\mathrm{eff}}^{\mathrm{SGD}} - W_{\mathrm{eff}}^{\mathrm{BL}}\|_F"
                    r" / \|W_{\mathrm{eff}}^{\mathrm{BL}}\|_F$"
                )

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
        f" | width={width} | B={batch_size} | {n_runs_any} runs"
        f" | Online (infinite data)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


_seed_overlay_ctx: dict = {}


def _init_seed_overlay_worker(
    precomputed: dict, output_dir: str,
) -> None:
    _seed_overlay_ctx["data"] = precomputed
    _seed_overlay_ctx["output_dir"] = output_dir


def _run_seed_overlay_task(task: tuple) -> None:
    gamma, noise, width, batch_size, model_seeds, filename = task
    data = _seed_overlay_ctx["data"]
    output_dir = Path(_seed_overlay_ctx["output_dir"])

    fig = plot_seed_overlay(data, gamma, noise, width, batch_size, model_seeds)
    if fig is None:
        return

    fig.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_seed_overlay_plots(
    input_dir: Path,
    bl_fallback: dict,
    output_dir: Path,
) -> None:
    precomputed = _precompute_seed_overlay_data(input_dir, bl_fallback)
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
        initargs=(precomputed, str(output_dir)),
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


def _init_plot_worker(all_stats: dict, output_dir: str) -> None:
    _worker_ctx["stats"] = all_stats
    _worker_ctx["output_dir"] = output_dir


def _run_plot_task(task: tuple) -> None:
    plot_type, gamma, noise, width, batch_size, model_seeds, filename = task
    stats = _worker_ctx["stats"]
    output_dir = Path(_worker_ctx["output_dir"])

    if plot_type == "distances":
        fig = plot_distances(stats, gamma, noise, width, batch_size, model_seeds)
    elif plot_type == "model_metrics":
        fig = plot_model_metrics(stats, gamma, noise, width, batch_size, model_seeds)
    else:
        return

    if fig is None:
        return

    fig.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    stats: dict[tuple, CompConfigStats],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    widths = sorted({k[0] for k in stats})
    gammas = sorted({k[1] for k in stats})
    noise_levels = sorted({k[2] for k in stats})
    model_seeds = sorted({k[3] for k in stats})
    batch_sizes = sorted({k[4] for k in stats})

    has_sgd_metrics = any(len(cs.sgd_layer_norms) > 0 for cs in stats.values())

    tasks = []
    for gamma in gammas:
        for noise in noise_levels:
            for width in widths:
                tag = f"g{gamma}_noise{noise}_w{width}"
                tasks.append(("distances", gamma, noise, width,
                              batch_sizes, model_seeds,
                              f"distances_{tag}"))

                if has_sgd_metrics:
                    for batch_size in batch_sizes:
                        tag_b = f"g{gamma}_noise{noise}_w{width}_b{batch_size}"
                        common = (gamma, noise, width, batch_size, model_seeds)
                        tasks.append(("model_metrics", *common, f"model_metrics_{tag_b}"))

    if not has_sgd_metrics:
        print("  Note: SGD model metrics not available — skipping model_metrics figures.")

    n_workers = min(os.cpu_count() or 1, len(tasks))
    print(f"Generating {len(tasks)} figures across {n_workers} workers...")

    with Pool(
        n_workers,
        initializer=_init_plot_worker,
        initargs=(stats, str(output_dir)),
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
        description="Plot online large-batch vs SGD comparative metrics",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Comparative results directory",
    )
    parser.add_argument(
        "--baseline-input", type=Path, default=DEFAULT_BASELINE_INPUT,
        help="Baseline (B=500) model metrics directory",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output figures directory",
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help="Cache file for computed statistics",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Force recompute statistics (ignore cache)",
    )
    args = parser.parse_args()

    print(f"Input:          {args.input}")
    print(f"Baseline input: {args.baseline_input}")
    print(f"Output:         {args.output}")

    bl_fallback = _load_baseline_metrics(args.baseline_input)

    stats = get_stats(
        args.input, bl_fallback,
        args.cache, args.recompute,
    )
    generate_all_plots(stats, args.output)
    generate_seed_overlay_plots(args.input, bl_fallback, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
